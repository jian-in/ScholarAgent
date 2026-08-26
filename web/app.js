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
let selectedMode = "auto";
let pollTimer = null;
let activeJobId = null;
let slowWarned = false;
let lastAnswerText = "";

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

function resetProgress() {
  if (progressLog) {
    progressLog.innerHTML = "";
    progressLog.hidden = true;
  }
  slowWarned = false;
  setTimeoutHint(false);
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

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    document.querySelector("#connection-label").textContent = "本地服务已连接";
    document.querySelector(".status-dot").classList.add("ready");
    document.querySelector("#model-state").textContent =
      `${status.model} · ${status.policy_available ? "学习策略可用" : "规则路由待命"}`;
  } catch (error) {
    document.querySelector("#connection-label").textContent = "本地服务不可用";
    document.querySelector("#model-state").textContent = "无法连接本地服务";
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

function renderArtifacts(artifacts) {
  if (!artifactsPanel || !artifactsBody) return;
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
  if (artifacts.papers && artifacts.papers.length) {
    const items = artifacts.papers.map((paper) => {
      const path = paper.path ? `<span class="artifact-path">${escapeHtml(paper.path)}</span>` : "";
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
      const path = note.path ? `<span class="artifact-path">${escapeHtml(note.path)}</span>` : "";
      const summary = note.summary ? ` — ${escapeHtml(note.summary)}` : "";
      return `<li><strong>${escapeHtml(note.title || "笔记")}</strong>${summary}${path}</li>`;
    }).join("");
    blocks.push(`<div class="artifact-group"><h3>研究笔记 · ${artifacts.notes.length}</h3><ul>${items}</ul></div>`);
  }
  if (artifacts.memories && artifacts.memories.length) {
    const items = artifacts.memories.map((memory) => {
      const source = memory.source ? `（${escapeHtml(memory.source)}）` : "";
      const path = memory.path ? `<span class="artifact-path">${escapeHtml(memory.path)}</span>` : "";
      return `<li>${escapeHtml(memory.text || "")}${source}${path}</li>`;
    }).join("");
    blocks.push(`<div class="artifact-group"><h3>长期记忆 · ${artifacts.memories.length}</h3><ul>${items}</ul></div>`);
  }

  artifactsPanel.hidden = false;
  artifactsBody.innerHTML = blocks.join("");
}

function renderResult(result) {
  renderRouting(result);
  renderArtifacts(result.artifacts);
  const metrics = result.metrics;
  const metricText = metrics
    ? `LLM ${metrics.llm_calls} 次 · 工具 ${metrics.tool_calls} 次`
    : "指标未采集";
  lastAnswerText = result.answer || "";
  const hasHtml = Boolean(result.answer_html);
  const answerClass = hasHtml ? "result-answer md" : "result-answer plain";
  const answerInner = hasHtml
    ? result.answer_html
    : escapeHtml(result.answer || "");
  resultBody.innerHTML =
    `<div class="result-meta">` +
    `<span>${escapeHtml((result.mode || "").toUpperCase())}</span>` +
    `<span>${escapeHtml(metricText)}</span>` +
    `</div>` +
    `<div class="${answerClass}">${answerInner}</div>`;
  setCopyVisible(Boolean(lastAnswerText));
}

function finishUi() {
  activeJobId = null;
  setCancelVisible(false);
  runButton.disabled = false;
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
      renderResult(payload);
      setRunState("已完成");
      setTimeoutHint(false);
    } else if (payload.status === "cancelled") {
      renderResult(payload);
      setRunState("已取消", "cancelled");
      setTimeoutHint(true, "任务已在下一步边界停止。");
    } else {
      resultBody.innerHTML = `<p>${escapeHtml(payload.error || "执行失败")}</p>`;
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
  setCopyVisible(false);
  runButton.disabled = true;
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

document.querySelectorAll("[data-mode]").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});
document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    taskInput.value = button.dataset.example;
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
taskInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") runTask();
});
loadStatus();
