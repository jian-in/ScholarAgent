"""声明式科研工作流与来源格式识别的行为契约。"""

from scholaragent.workflow import WorkflowRegistry, detect_source_format
from scholaragent.evidence import EvidenceLedger
from scholaragent.llm import ScriptedLLM
from scholaragent.runtime import create_runtime
from scholaragent.tool import Tool, ToolResult
from scholaragent.tools.papers import ReadPaperTool
from scholaragent.workspace import TemporaryWorkspace


def test_registry_selects_source_aware_workflows_without_executing_code():
    registry = WorkflowRegistry.default()

    reader = registry.select(
        "把这篇论文做成中英文对照全文 reader",
        source="paper.pdf",
    )
    assert reader.name == "paper-reading"
    assert reader.source_format == "pdf-text"
    assert "source_map.json" in reader.spec.outputs
    assert reader.spec.validators

    card = registry.select("生成 Paper Card 并做结论证据链审查")
    assert card.name == "paper-card"
    assert card.spec.evidence_requirements

    assert detect_source_format("10.48550/arXiv.2401.12345") == "doi-arxiv"
    assert detect_source_format("https://example.org/preprint.html") == "html"


def test_evidence_ledger_keeps_claims_tied_to_stable_source_anchors():
    ledger = EvidenceLedger()
    ledger.add_anchor(
        id="S001",
        kind="text",
        source="arxiv:2401.12345",
        page=3,
        section="Methods",
        confidence="high",
    )
    ledger.add_claim(
        id="C001",
        claim="该方法使用残差连接。",
        anchor_ids=["S001"],
    )

    assert ledger.validate() == []
    data = ledger.to_dict()
    assert data["schema_version"] == "evidence-ledger-v1"
    assert data["summary"]["supported_claims"] == 1
    assert data["claims"][0]["anchor_ids"] == ["S001"]


def test_runtime_records_workflow_and_source_evidence_without_changing_runner_semantics(tmp_path):
    runtime = create_runtime(
        llm=ScriptedLLM([{
            "content": "结论",
            "tool_calls": [],
        }]),
        workspace=TemporaryWorkspace(tmp_path),
        conversation=False,
        auto_recall=False,
    )

    result = runtime.run(
        "生成 Paper Card 并做结论证据链审查",
        mode="react",
        source="10.48550/arXiv.2401.12345",
    )

    assert result.status == "completed"
    assert result.workflow == "paper-card"
    assert result.source_format == "doi-arxiv"
    assert any(event["type"] == "workflow_selected" for event in result.events)
    assert result.evidence["schema_version"] == "evidence-ledger-v1"


def test_tool_source_anchor_flows_into_run_result_evidence(tmp_path):
    class AnchoredTool(Tool):
        name = "anchored"
        description = "返回带来源锚点的离线证据"
        parameters = {"type": "object", "properties": {}, "required": []}

        def run_result(self, **kwargs):
            return ToolResult(
                text="已找到来源",
                artifacts=(
                    {
                        "kind": "read",
                        "source_anchors": [
                            {
                                "id": "S001",
                                "kind": "text",
                                "source": "demo-paper",
                                "page": 2,
                                "confidence": "high",
                            }
                        ],
                    },
                ),
            )

    runtime = create_runtime(
        llm=ScriptedLLM([
            {
                "content": "检索来源",
                "tool_calls": [{"id": "a1", "name": "anchored", "arguments": {}}],
            },
            {"content": "结论有来源", "tool_calls": []},
        ]),
        workspace=TemporaryWorkspace(tmp_path),
        conversation=False,
        auto_recall=False,
    )
    runtime.registry.register(AnchoredTool())

    result = runtime.run("一个离线任务", mode="react")

    assert result.evidence["summary"]["anchors"] == 1
    assert result.evidence["anchors"][0]["id"] == "S001"


def test_read_paper_emits_page_anchors_without_changing_model_text():
    tool = ReadPaperTool()
    tool.start_run()
    original = "--- 第 2 页 ---\n方法内容\n--- 第 3 页 ---\n实验内容\n(全文读完)"

    metadata = tool.artifact_metadata(
        {"arxiv_id": "2401.12345"},
        ToolResult(text=original),
    )

    assert metadata[0]["kind"] == "read"
    assert [anchor["id"] for anchor in metadata[0]["source_anchors"]] == ["S001", "S002"]
    assert metadata[0]["source_anchors"][0]["page"] == 2
