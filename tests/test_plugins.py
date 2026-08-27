"""工具插件发现的公开行为。"""

from scholaragent.tool import Tool, ToolRegistry
from scholaragent.tools.calculator import CalculatorTool


class FakeEntryPoint:
    group = "scholaragent.tools"

    def __init__(self, name, loader):
        self.name = name
        self._loader = loader

    def load(self):
        return self._loader()


class WordCountTool(Tool):
    name = "word_count"
    description = "统计文本中的词数"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    license = "MIT"

    def run(self, text=""):
        return str(len(str(text).split()))


def test_external_plugin_loads_without_changing_builtin_registration():
    registry = ToolRegistry([CalculatorTool()])
    report = registry.discover_plugins(
        entry_points=[FakeEntryPoint("word-count", WordCountTool)]
    )

    assert report == {"loaded": ["word_count"], "errors": []}
    assert {item["function"]["name"] for item in registry.schemas()} == {
        "calculator", "word_count",
    }
    assert registry.call("word_count", {"text": "one two three"}) == "3"


def test_bad_and_duplicate_plugins_are_isolated_and_diagnosable():
    class NoLicense(Tool):
        name = "no_license"
        description = "invalid"

        def run(self, **kwargs):
            return "never"

        license = ""

    class UndeclaredLicense(Tool):
        name = "undeclared_license"
        description = "invalid"

        def run(self, **kwargs):
            return "never"

    registry = ToolRegistry([WordCountTool()])
    report = registry.discover_plugins(entry_points=[
        FakeEntryPoint("duplicate", WordCountTool),
        FakeEntryPoint("no-license", NoLicense),
        FakeEntryPoint("undeclared-license", UndeclaredLicense),
        FakeEntryPoint("broken", lambda: (_ for _ in ()).throw(RuntimeError("坏插件"))),
    ])

    assert report["loaded"] == []
    assert {item["name"] for item in report["errors"]} == {
        "duplicate", "no-license", "broken",
        "undeclared-license",
    }
    assert registry.call("word_count", {"text": "still works"}) == "2"
