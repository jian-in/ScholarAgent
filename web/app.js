const taskInput = document.querySelector("#task-input");
const runButton = document.querySelector("#run-task");
const cancelButton = document.querySelector("#cancel-task");
const runState = document.querySelector("#run-state");
const resultBody = document.querySelector("#result-body");
const progressLog = document.querySelector("#progress-log");
const timeoutHint = document.querySelector("#timeout-hint");
const actionRow = document.querySelector(".action-row");
const copyButton = document.querySelector("#copy-answer");
const artifactsPanel = document.querySelector("#artifacts-panel");
const artifactsBody = document.querySelector("#artifacts-body");
const caseSelector = document.querySelector("#case-selector");
const replayButton = document.querySelector("#replay-case");
const savedCaseState = document.querySelector("#saved-case-state");
const timelinePanel = document.querySelector("#event-timeline");
const timelineBody = document.querySelector("#timeline-body");
const modelSelector = document.querySelector("#model-selector");
const applyModelButton = document.querySelector("#apply-model");
const modelHint = document.querySelector("#model-hint");
const singleRoutingButton = document.querySelector("#single-routing");
const splitRoutingButton = document.querySelector("#split-routing");
const routingHint = document.querySelector("#routing-hint");
let selectedMode = "auto";
let pollTimer = null;
let activeJobId = null;
let slowWarned = false;
let lastAnswerText = "";
let liveEvents = [];
let modelCatalogReady = false;
let modelRoutingState = null;

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = value == null ? "" : String(value);
  return element.innerHTML;
}

function setRunState(label, state = "") {
  runState.textContent = label;
  runState.className = `run-state ${state}`;
}

function setMode(mode) {
  selectedMode = mode;
  document.querySelectorAll("[data-mode]").forEach((button) => {
    const selected = button.dataset.mode === mode;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-checked", String(selected));
  });
}

function clearPoll() {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function setCancelVisible(visible) {
  if (!cancelButton) return;
  cancelButton.hidden = !visible;
  cancelButton.disabled = false;
  if (actionRow) actionRow.classList.toggle("has-cancel", visible);
}

function setTimeoutHint(visible, text) {
  if (!timeoutHint) return;
  timeoutHint.hidden = !visible;
  if (text) timeoutHint.textContent = text;
}

function setModelControlsDisabled(disabled) {
  if (modelSelector) modelSelector.disabled = disabled || !modelCatalogReady;
  if (applyModelButton) {
    applyModelButton.disabled = disabled || !modelCatalogReady || !(modelSelector && modelSelector.value);
  }
  if (singleRoutingButton) singleRoutingButton.disabled = disabled || !modelRoutingState;
  if (splitRoutingButton) splitRoutingButton.disabled = disabled || !modelRoutingState;
}

function resetProgress() {
  if (progressLog) {
    progressLog.innerHTML = "";
    progressLog.hidden = true;
  }
  slowWarned = false;
  setTimeoutHint(false);
}

function resetTimeline() {
  if (timelinePanel) timelinePanel.hidden = true;
  if (timelineBody) timelineBody.innerHTML = "";
}

function renderTimeline(events) {
  if (!timelinePanel || !timelineBody) return;
  if (!events || !events.length) {
    resetTimeline();
    return;
  }
  const labels = {
    run_started: "运行开始",
    mode_selected: "模式选择",
    model_step: "模型步骤",
    tool_started: "工具开始",
    tool_completed: "工具完成",
    artifact: "产物",
    cancel_requested: "请求取消",
    cancelled: "已取消",
    failed: "失败",
    completed: "已完成",
    legacy_progress: "历史进度",
  };
  timelineBody.innerHTML = events.map((event) => {
    const payload = event.payload || {};
    const detail = payload.message || payload.name || payload.reason ||
      (event.type === "model_step"
        ? `第 ${payload.step ?? "—"} 步 · ${payload.role || "模型"}` +
          (payload.provider || payload.model
            ? ` · ${payload.provider || ""}/${payload.model || ""}`
            : "")
        : "");
    const timestamp = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : "历史";
    return `<div class="timeline-entry">` +
      `<time>${escapeHtml(timestamp)}</time>` +
      `<strong>${escapeHtml(labels[event.type] || event.type)}</strong>` +
      `<span>${escapeHtml(detail)}</span>` +
      `</div>`;
  }).join("");
  timelinePanel.hidden = false;
}

function appendLogs(logs) {
  if (!progressLog || !logs || !logs.length) return;
  progressLog.hidden = false;
  const fragment = document.createDocumentFragment();
  for (const entry of logs) {
    const line = document.createElement("div");
    line.className = "progress-line";
    const message = entry.message || "";
    if (message.includes("调用工具")) line.classList.add("tool");
    else if (message.includes("工具返回")) line.classList.add("observation");
    else if (
      message.startsWith("[团队]") ||
      message.startsWith("[计划]") ||
      message.startsWith("[执行") ||
      message.startsWith("[反思]") ||
      message.startsWith("[取消]") ||
      message.startsWith("[超时]")
    ) {
      line.classList.add("stage");
    }
    line.textContent = message;
    fragment.appendChild(line);
  }
  progressLog.appendChild(fragment);
  progressLog.scrollTop = progressLog.scrollHeight;
}

function mergeEvents(events) {
  const incoming = events || [];
  const seen = new Set(liveEvents.map((event) =>
    `${event.run_id || ""}|${event.type || ""}|${event.timestamp || ""}|${JSON.stringify(event.payload || {})}`
  ));
  for (const event of incoming) {
    const key = `${event.run_id || ""}|${event.type || ""}|${event.timestamp || ""}|${JSON.stringify(event.payload || {})}`;
    if (!seen.has(key)) {
      seen.add(key);
      liveEvents.push(event);
    }
  }
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    document.querySelector("#connection-label").textContent = "本地服务已连接";
    document.querySelector(".status-dot").classList.add("ready");
    const ocrLabel = status.ocr && status.ocr.available ? "OCR 可用" : "OCR 待配置";
    const routing = status.model_routing;
    if (routing && routing.mode === "split") {
      const research = routing.roles && routing.roles.research;
      const summary = routing.roles && routing.roles.summary;
      document.querySelector("#model-state").textContent =
        `双模型 · 调研 ${research ? `${research.provider}/${research.model}` : "云端"}` +
        ` · 总结 ${summary ? `${summary.provider}/${summary.model}` : "本地"}` +
        ` · ${status.policy_available ? "学习策略可用" : "规则路由待命"} · ${ocrLabel}`;
    } else {
      const providerLabel = status.provider === "ollama" ? "本地 Ollama" : "云端接口";
      document.querySelector("#model-state").textContent =
        `${providerLabel} · ${status.model} · ${status.policy_available ? "学习策略可用" : "规则路由待命"} · ${ocrLabel}`;
    }
  } catch (error) {
    document.querySelector("#connection-label").textContent = "本地服务不可用";
    document.querySelector("#model-state").textContent = "无法连接本地服务";
  }
}

function formatModelOption(item) {
  const parts = [item.label || item.model || "未命名模型"];
  if (item.size_bytes) {
    parts.push(`${(Number(item.size_bytes) / 1024 / 1024 / 1024).toFixed(1)} GB`);
  }
  if (item.context_length) {
    parts.push(`${Math.round(Number(item.context_length) / 1024)}K 上下文`);
  }
  if (item.supports_tools === false) parts.push("不支持工具调用");
  return parts.join(" · ");
}

function renderModelCatalog(catalog) {
  if (!modelSelector) return;
  modelSelector.innerHTML = "";
  const options = catalog.options || [];
  for (const item of options) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = formatModelOption(item);
    option.disabled = item.supports_tools === false;
    modelSelector.appendChild(option);
  }
  const currentId = catalog.current && catalog.current.id;
  if (currentId && options.some((item) => item.id === currentId)) {
    modelSelector.value = currentId;
  } else if (!options.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = catalog.ollama_available
      ? "没有可用的工具调用模型"
      : "未发现本地 Ollama 模型";
    modelSelector.appendChild(option);
  }
  modelCatalogReady = options.length > 0;
  if (modelHint) {
    modelHint.textContent = catalog.cloud_configured
      ? "可在本地模型与已配置云端档案之间切换；只影响新任务。"
      : "当前仅发现本地模型；云端档案需在 .env 中配置。";
  }
  renderModelRouting(catalog.routing);
  setModelControlsDisabled(Boolean(activeJobId));
}

function renderModelRouting(routing) {
  modelRoutingState = routing || null;
  const mode = routing && routing.mode;
  if (singleRoutingButton) singleRoutingButton.classList.toggle("selected", mode === "single");
  if (splitRoutingButton) splitRoutingButton.classList.toggle("selected", mode === "split");
  if (!routingHint) return;
  if (!routing) {
    routingHint.textContent = "模型分工状态不可用。";
    return;
  }
  const research = routing.roles && routing.roles.research;
  const summary = routing.roles && routing.roles.summary;
  const timeout = routing.soft_timeout_seconds
    ? ` · 软超时 ${routing.soft_timeout_seconds}s（可配置）`
    : "";
  if (mode === "split") {
    routingHint.textContent =
      `当前：调研 ${research ? `${research.provider}/${research.model}` : "云端"}` +
      ` · 总结 ${summary ? `${summary.provider}/${summary.model}` : "本地"}${timeout}`;
  } else if (routing.requested_mode === "split" && !routing.available) {
    routingHint.textContent = `双模型暂不可用：${routing.reason}`;
  } else {
    routingHint.textContent = `当前任务由同一个模型负责调研与总结。${timeout}`;
  }
}

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    const catalog = await response.json();
    if (!response.ok) throw new Error(catalog.error || "模型目录加载失败");
    renderModelCatalog(catalog);
  } catch (error) {
    modelCatalogReady = false;
    if (modelSelector) {
      modelSelector.innerHTML = '<option value="">模型目录不可用</option>';
      modelSelector.disabled = true;
    }
    if (applyModelButton) applyModelButton.disabled = true;
    if (modelHint) modelHint.textContent = error.message || "模型目录不可用。";
  }
}

async function applyModel() {
  if (!modelSelector || !modelSelector.value || activeJobId) return;
  const selected = modelSelector.options[modelSelector.selectedIndex];
  const [provider, ...modelParts] = modelSelector.value.split("|");
  const model = provider === "ollama" ? modelParts.join("|") : "";
  applyModelButton.disabled = true;
  modelSelector.disabled = true;
  if (modelHint) modelHint.textContent = "正在切换模型…";
  try {
    const response = await fetch("/api/model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider, model }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "模型切换失败");
    renderModelCatalog(payload);
    await loadStatus();
    if (modelHint) modelHint.textContent = `${selected.textContent} 已生效，只影响新任务。`;
  } catch (error) {
    if (modelHint) modelHint.textContent = error.message || "模型切换失败。";
    setModelControlsDisabled(false);
  }
}

async function applyRouting(mode) {
  if (activeJobId || !modelRoutingState) return;
  setModelControlsDisabled(true);
  if (routingHint) routingHint.textContent = "正在切换任务模型分工…";
  try {
    const response = await fetch("/api/model-routing", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "模型分工切换失败");
    renderModelCatalog(payload);
    await loadStatus();
  } catch (error) {
    if (routingHint) routingHint.textContent = error.message || "模型分工切换失败。";
    setModelControlsDisabled(false);
  }
}

async function loadCases() {
  if (!caseSelector) return;
  try {
    const response = await fetch("/api/cases");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "案例加载失败");
    const cases = payload.cases || [];
    caseSelector.innerHTML = cases.length
      ? cases.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.title)} · ${item.runs} 次运行</option>`).join("")
      : '<option value="">暂无公开案例</option>';
    if (replayButton) replayButton.disabled = !cases.length;
    if (savedCaseState) savedCaseState.textContent = cases.length
      ? "选择案例后可在同一界面回放；历史结果会标注评分状态。"
      : "当前没有可回放的公开案例。";
  } catch (error) {
    if (caseSelector) caseSelector.innerHTML = '<option value="">案例服务不可用</option>';
    if (replayButton) replayButton.disabled = true;
    if (savedCaseState) savedCaseState.textContent = "无法加载已保存案例。";
  }
}

function renderRouting(result) {
  const route = result.routing;
  document.querySelector("#route-mode").textContent = (result.mode || "—").toUpperCase();
  document.querySelector("#elapsed-time").textContent =
    result.seconds == null ? "—" : `${Number(result.seconds).toFixed(2)} s`;
  if (!route) {
    document.querySelector("#route-reason").textContent = "已按显式模式执行。";
    document.querySelector("#policy-version").textContent = "显式模式";
    document.querySelector("#route-utility").textContent = "不适用";
    return;
  }
  document.querySelector("#route-reason").textContent = route.reason;
  document.querySelector("#policy-version").textContent = route.policy_version;
  const utility = route.predicted_utility && route.predicted_utility[result.mode];
  document.querySelector("#route-utility").textContent =
    utility == null ? "—" : Number(utility).toFixed(2);
}

function setCopyVisible(visible) {
  if (!copyButton) return;
  copyButton.hidden = !visible;
  if (visible) copyButton.textContent = "复制回答";
}

function resetArtifacts() {
  if (artifactsPanel) artifactsPanel.hidden = true;
  if (artifactsBody) artifactsBody.innerHTML = "";
}

function copyText(text, btn) {
  const done = () => {
    if (!btn) return;
    const original = btn.textContent;
    btn.textContent = "已复制";
    setTimeout(() => { btn.textContent = original; }, 1200);
  };
  const fallback = () => {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    try { document.execCommand("copy"); done(); } finally {
      document.body.removeChild(textarea);
    }
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done).catch(fallback);
  } else {
    fallback();
  }
}

let artifactCopyBound = false;

function bindArtifactCopy() {
  if (artifactCopyBound || !artifactsBody) return;
  artifactCopyBound = true;
  artifactsBody.addEventListener("click", (event) => {
    const target = event.target;
    if (target.classList.contains("artifact-copy")) {
      copyText(target.dataset.copy || "", target);
      return;
    }
    const pathSpan = target.closest(".artifact-path");
    if (pathSpan && pathSpan.dataset.copy) {
      copyText(pathSpan.dataset.copy, null);
      pathSpan.title = "已复制";
      setTimeout(() => { pathSpan.title = "点击复制完整路径"; }, 1200);
    }
  });
}

function renderArtifacts(artifacts) {
  if (!artifactsPanel || !artifactsBody) return;
  bindArtifactCopy();
  if (!artifacts) {
    resetArtifacts();
    return;
  }
  const counts = artifacts.counts || {};
  const total =
    (counts.papers || 0) +
    (counts.notes || 0) +
    (counts.memories || 0) +
    (counts.read || 0);
  if (!total) {
    artifactsPanel.hidden = false;
    artifactsBody.innerHTML = '<p class="artifact-empty">本轮未落盘论文、笔记或长期记忆。</p>';
    return;
  }

  const blocks = [];
  if (artifacts.root) {
    const safeRoot = escapeHtml(artifacts.root).replace(/"/g, "&quot;");
    blocks.push(
      `<div class="artifact-root"><strong>本轮产物目录</strong>` +
      `<span class="artifact-path" data-copy="${safeRoot}" title="点击复制完整路径">${escapeHtml(artifacts.root)}</span>` +
      `<button type="button" class="artifact-copy" data-copy="${safeRoot}">复制路径</button></div>`
    );
  }
  if (artifacts.papers && artifacts.papers.length) {
    const items = artifacts.papers.map((paper) => {
      const path = paper.path
        ? `<span class="artifact-path" data-copy="${escapeHtml(paper.path).replace(/"/g, "&quot;")}" title="点击复制完整路径">${escapeHtml(paper.path)}</span>`
        : "";
      return `<li><strong>${escapeHtml(paper.arxiv_id || "论文")}</strong>${path}</li>`;
    }).join("");
    blocks.push(`<div class="artifact-group"><h3>下载论文 · ${artifacts.papers.length}</h3><ul>${items}</ul></div>`);
  }
  if (artifacts.read_ids && artifacts.read_ids.length) {
    const items = artifacts.read_ids.map((id) => `<li>${escapeHtml(id)}</li>`).join("");
    blocks.push(`<div class="artifact-group"><h3>已阅读 · ${artifacts.read_ids.length}</h3><ul>${items}</ul></div>`);
  }
  if (artifacts.notes && artifacts.notes.length) {
    const items = artifacts.notes.map((note) => {
      const path = note.path
        ? `<span class="artifact-path" data-copy="${escapeHtml(note.path).replace(/"/g, "&quot;")}" title="点击复制完整路径">${escapeHtml(note.path)}</span>`
        : "";
      const summary = note.summary ? ` — ${escapeHtml(note.summary)}` : "";
      return `<li><strong>${escapeHtml(note.title || "笔记")}</strong>${summary}${path}</li>`;
    }).join("");
    blocks.push(`<div class="artifact-group"><h3>研究笔记 · ${artifacts.notes.length}</h3><ul>${items}</ul></div>`);
  }
  if (artifacts.memories && artifacts.memories.length) {
    const items = artifacts.memories.map((memory) => {
      const source = memory.source ? `（${escapeHtml(memory.source)}）` : "";
      const path = memory.path
        ? `<span class="artifact-path" data-copy="${escapeHtml(memory.path).replace(/"/g, "&quot;")}" title="点击复制完整路径">${escapeHtml(memory.path)}</span>`
        : "";
      return `<li>${escapeHtml(memory.text || "")}${source}${path}</li>`;
    }).join("");
    blocks.push(`<div class="artifact-group"><h3>长期记忆 · ${artifacts.memories.length}</h3><ul>${items}</ul></div>`);
  }

  artifactsPanel.hidden = false;
  artifactsBody.innerHTML = blocks.join("");
}

function formatTokenUsage(metrics) {
  if (!metrics || metrics.prompt_tokens == null || metrics.completion_tokens == null) {
    return "Token 未返回";
  }
  const promptTokens = Number(metrics.prompt_tokens);
  const completionTokens = Number(metrics.completion_tokens);
  if (!Number.isFinite(promptTokens) || !Number.isFinite(completionTokens)) {
    return "Token 未返回";
  }
  const totalTokens = promptTokens + completionTokens;
  let cacheText = "";
  if (metrics.cache_hit_tokens != null && metrics.cache_miss_tokens != null) {
    const hit = Number(metrics.cache_hit_tokens);
    const totalPrompt = hit + Number(metrics.cache_miss_tokens);
    if (Number.isFinite(hit) && totalPrompt > 0) {
      cacheText = ` · 缓存命中 ${(Number.isInteger(hit / totalPrompt * 100) ? hit / totalPrompt * 100 : (hit / totalPrompt * 100).toFixed(1))}%（${hit.toLocaleString()}）`;
    }
  }
  return `Token ${totalTokens.toLocaleString()}（输入 ${promptTokens.toLocaleString()} · 输出 ${completionTokens.toLocaleString()}）${cacheText}`;
}

function formatRoleTokenUsage(metrics) {
  const groups = metrics && metrics.llm_usage_by_role;
  if (!groups || typeof groups !== "object") return "";
  const labels = { research: "调研", summary: "总结", general: "模型" };
  return Object.entries(groups).map(([role, item]) => {
    const provider = item.provider ? `${item.provider}/` : "";
    const model = item.model || "";
    const tokens = item.prompt_tokens == null || item.completion_tokens == null
      ? "Token 未返回"
      : `${(Number(item.prompt_tokens) + Number(item.completion_tokens)).toLocaleString()} token`;
    return `${labels[role] || role} ${provider}${model} · ${tokens}`;
  }).join(" · ");
}

function renderResult(result, source = "realtime", eventOverride = null) {
  renderRouting(result);
  renderArtifacts(result.artifacts);
  renderTimeline(eventOverride || result.events || []);
  const metrics = result.metrics;
  const tokenText = formatTokenUsage(metrics);
  const roleTokenText = formatRoleTokenUsage(metrics);
  const metricText = metrics
    ? `LLM ${metrics.llm_calls} 次 · 工具 ${metrics.tool_calls} 次 · ${tokenText}` +
      (roleTokenText ? ` · ${roleTokenText}` : "")
    : "指标未采集";
  lastAnswerText = result.answer || "";
  const hasHtml = Boolean(result.answer_html);
  const answerClass = hasHtml ? "result-answer md" : "result-answer plain";
  const answerInner = hasHtml
    ? result.answer_html
    : escapeHtml(result.answer || "");
  const sourceLabel = source === "saved_case" ? "已保存案例" : "实时运行";
  const scoreLabels = {
    unscored: "未评分",
    author_scored: "作者评分",
    independently_scored: "独立评分",
  };
  const sourceLabels = {
    "task-only": "任务文本",
    "pdf-text": "PDF 文本",
    "scanned-pdf": "扫描 PDF",
    html: "HTML",
    "doi-arxiv": "DOI/arXiv",
    "pasted-text": "粘贴文本",
  };
  const evidenceSummary = result.evidence && result.evidence.summary;
  const evidenceLabel = evidenceSummary
    ? `证据锚点 ${evidenceSummary.anchors || 0} · 校验 ${evidenceSummary.validation_errors ? "有误" : "通过"}`
    : "";
  const statusLabel = result.status === "failed"
    ? "失败"
    : result.status === "cancelled" ? "已取消" : "完成";
  resultBody.innerHTML =
    `<div class="result-meta">` +
    `<span>${escapeHtml(sourceLabel)}</span>` +
    `<span>${escapeHtml((result.mode || "").toUpperCase())}</span>` +
    `<span>${escapeHtml(metricText)}</span>` +
    `<span>${escapeHtml(statusLabel)}</span>` +
    (result.workflow ? `<span>工作流 ${escapeHtml(result.workflow)}</span>` : "") +
    (result.model_routing ? `<span>模型分工 ${escapeHtml(result.model_routing.mode === "split" ? "云端调研 · 本地总结" : "单模型")}</span>` : "") +
    (result.source_format ? `<span>来源 ${escapeHtml(sourceLabels[result.source_format] || result.source_format)}</span>` : "") +
    (evidenceLabel ? `<span>${escapeHtml(evidenceLabel)}</span>` : "") +
    (result.score_state ? `<span>${escapeHtml(scoreLabels[result.score_state] || result.score_state)}</span>` : "") +
    `</div>` +
    (result.error ? `<p class="result-error">${escapeHtml(result.error)}</p>` : "") +
    `<div class="${answerClass}">${answerInner}</div>`;
  setCopyVisible(Boolean(lastAnswerText));
}

function finishUi() {
  activeJobId = null;
  setCancelVisible(false);
  runButton.disabled = false;
  setModelControlsDisabled(false);
}

async function copyAnswer() {
  if (!lastAnswerText) return;
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(lastAnswerText);
    } else {
      const area = document.createElement("textarea");
      area.value = lastAnswerText;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.left = "-9999px";
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      document.body.removeChild(area);
    }
    if (copyButton) {
      copyButton.textContent = "已复制";
      setTimeout(() => {
        if (copyButton) copyButton.textContent = "复制回答";
      }, 1500);
    }
  } catch (error) {
    if (copyButton) copyButton.textContent = "复制失败";
  }
}

async function pollJob(jobId, after = 0) {
  try {
    const response = await fetch(`/api/jobs/${jobId}?after=${after}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "查询任务失败");

    appendLogs(payload.logs || []);
    mergeEvents(payload.all_events || payload.events);
    const nextAfter = payload.next_after ?? after;
    const elapsed = Number(payload.elapsed || 0);

    if (payload.status === "running" || payload.status === "queued") {
      if (payload.cancel_requested) {
        setRunState(`正在取消 · 已运行 ${elapsed.toFixed(0)}s`, "slow");
        setTimeoutHint(true, "已请求取消，将在当前模型/工具步骤结束后停止。");
      } else if (payload.slow) {
        if (!slowWarned) {
          slowWarned = true;
          setTimeoutHint(
            true,
            `任务已运行 ${elapsed.toFixed(0)}s。Team/Plan 可能更久，可继续等待或点「取消任务」。`
          );
        }
        setRunState(`运行较久 · ${elapsed.toFixed(0)}s · ${payload.log_count || 0} 步`, "slow");
      } else {
        const logHint = payload.log_count
          ? `正在执行 · ${elapsed.toFixed(0)}s · ${payload.log_count} 步`
          : `正在执行 · ${elapsed.toFixed(0)}s`;
        setRunState(logHint, "running");
      }
      pollTimer = setTimeout(() => pollJob(jobId, nextAfter), 600);
      return;
    }

    if (payload.status === "done") {
      renderResult(payload, "realtime", liveEvents);
      setRunState("已完成");
      setTimeoutHint(false);
    } else if (payload.status === "cancelled") {
      renderResult(payload, "realtime", liveEvents);
      setRunState("已取消", "cancelled");
      setTimeoutHint(true, "任务已在下一步边界停止。");
    } else {
      renderResult(payload, "realtime", liveEvents);
      setRunState("执行失败", "error");
      setTimeoutHint(false);
    }
    finishUi();
  } catch (error) {
    resultBody.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
    setRunState("执行失败", "error");
    setTimeoutHint(false);
    finishUi();
  }
}

async function cancelTask() {
  if (!activeJobId) return;
  cancelButton.disabled = true;
  setRunState("正在请求取消…", "slow");
  try {
    const response = await fetch(`/api/jobs/${activeJobId}/cancel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "取消失败");
    setTimeoutHint(true, payload.message || "已请求取消。");
  } catch (error) {
    setTimeoutHint(true, error.message || "取消请求失败");
    cancelButton.disabled = false;
  }
}

async function runTask() {
  const task = taskInput.value.trim();
  if (!task) {
    taskInput.focus();
    setRunState("请输入任务", "error");
    return;
  }
  clearPoll();
  resetProgress();
  resetArtifacts();
  activeJobId = null;
  lastAnswerText = "";
  liveEvents = [];
  setCopyVisible(false);
  resetTimeline();
  if (savedCaseState) savedCaseState.textContent = "正在实时运行；完成后可继续回放公开案例。";
  runButton.disabled = true;
  setModelControlsDisabled(true);
  setCancelVisible(true);
  setRunState("正在提交", "running");
  resultBody.innerHTML = "<p>任务已提交，步骤日志会实时出现在下方；可随时取消。</p>";
  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, mode: selectedMode }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "提交失败");
    activeJobId = payload.job_id;
    setRunState("正在执行", "running");
    pollJob(payload.job_id, 0);
  } catch (error) {
    resultBody.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
    setRunState("执行失败", "error");
    finishUi();
  }
}

async function replayCase() {
  if (!caseSelector || !caseSelector.value) return;
  clearPoll();
  resetProgress();
  resetArtifacts();
  resetTimeline();
  try {
    const response = await fetch(`/api/cases/${encodeURIComponent(caseSelector.value)}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "案例回放失败");
    renderResult(payload.selected, "saved_case");
    setRunState("已回放", "");
    setTimeoutHint(false);
    if (savedCaseState) savedCaseState.textContent =
      `${payload.title} · 已保存证据，不会触发模型或网络请求。`;
  } catch (error) {
    resultBody.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
    setRunState("回放失败", "error");
  }
}

document.querySelectorAll("[data-mode]").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});
document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    taskInput.value = button.dataset.example;
    if (button.dataset.exampleMode) setMode(button.dataset.exampleMode);
    taskInput.focus();
  });
});
document.querySelector("#clear-task").addEventListener("click", () => {
  taskInput.value = "";
  taskInput.focus();
});
runButton.addEventListener("click", runTask);
if (cancelButton) cancelButton.addEventListener("click", cancelTask);
if (copyButton) copyButton.addEventListener("click", copyAnswer);
if (replayButton) replayButton.addEventListener("click", replayCase);
if (applyModelButton) applyModelButton.addEventListener("click", applyModel);
if (singleRoutingButton) singleRoutingButton.addEventListener("click", () => applyRouting("single"));
if (splitRoutingButton) splitRoutingButton.addEventListener("click", () => applyRouting("split"));
taskInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") runTask();
});
loadStatus();
loadModels();
loadCases();
