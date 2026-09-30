"""工具超时的离线回归：受控释放后台调用，不留下阻塞线程。"""

import threading
import time
import unittest

from scholaragent.tool import (
    DEFAULT_TOOL_TIMEOUT, STOP_RETRY_PREFIX, Tool, ToolRegistry, ToolResult,
)


class BlockingTool(Tool):
    name = "blocking"

    def __init__(self, name=None):
        if name:
            self.name = name
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.resets = 0
        self.metadata_calls = 0

    def run(self):
        self.calls += 1
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError("test failed to release background tool")
        return ToolResult("late result", artifacts=({"kind": "late"},))

    def start_run(self):
        self.resets += 1

    def artifact_metadata(self, arguments, result):
        self.metadata_calls += 1
        return [{"kind": "unexpected"}]


class QuickTool(Tool):
    name = "quick"

    def __init__(self):
        self.calls = 0
        self.thread = None

    def run(self):
        self.calls += 1
        self.thread = threading.get_ident()
        return "ok"


class ToolTimeoutTests(unittest.TestCase):
    def wait_for_idle(self, registry):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with registry._execution.lock:
                if not registry._execution.running:
                    self.assertIsNone(registry._execution.pool)
                    return
            time.sleep(0.001)
        self.fail("tool executor did not retire after tools completed")

    def test_default_timeout_and_lazy_pool(self):
        registry = ToolRegistry()
        self.assertEqual(registry.tool_timeout, DEFAULT_TOOL_TIMEOUT)
        self.assertIsNone(registry._execution.pool)
        self.assertFalse(registry.call_result("missing", {}).success)
        self.assertIsNone(registry._execution.pool)

    def test_disabled_timeout_preserves_synchronous_execution(self):
        for timeout in (0, -1):
            with self.subTest(timeout=timeout):
                tool = QuickTool()
                registry = ToolRegistry([tool], tool_timeout=timeout)
                self.assertEqual(registry.call("quick", {}), "ok")
                self.assertEqual(tool.thread, threading.get_ident())
                self.assertIsNone(registry._execution.pool)
                self.assertEqual(registry.subset(["quick"]).tool_timeout, timeout)

    def test_completed_structured_result_and_idle_pool_cleanup(self):
        tool = BlockingTool()
        tool.release.set()
        registry = ToolRegistry([tool], tool_timeout=1)
        result = registry.call_result(tool.name, {})
        self.assertTrue(result.success)
        self.assertEqual(result.artifacts, ({"kind": "late"},))
        self.assertEqual(registry.run_tool_calls, 1)
        self.wait_for_idle(registry)

    def test_timeout_is_bounded_fused_and_does_not_repeat_side_effects(self):
        tool, quick = BlockingTool(), QuickTool()
        registry = ToolRegistry([tool, quick], tool_timeout=0.1)
        try:
            before = time.monotonic()
            result = registry.call_result(tool.name, {})
            self.assertLess(time.monotonic() - before, 1)
            self.assertTrue(tool.entered.is_set())
            self.assertFalse(result.success)
            self.assertTrue(result.stop_retry)
            self.assertEqual(result.diagnostic["kind"], "timeout")
            self.assertIn("后台", result.text)
            self.assertEqual(result.artifacts, ())
            self.assertEqual(tool.metadata_calls, 0)
            self.assertEqual(registry.call_result(tool.name, {}).diagnostic["kind"],
                             "tool_busy")
            self.assertEqual(tool.calls, 1)
            self.assertEqual(registry.call("quick", {}), "ok")
        finally:
            tool.release.set()
            self.wait_for_idle(registry)

    def test_late_result_is_not_recorded_as_a_success(self):
        class Collector:
            def __init__(self):
                self.results = []

            def record_result(self, name, arguments, result):
                self.results.append(result)

        tool, collector = BlockingTool(), Collector()
        registry = ToolRegistry([tool], artifacts=collector, tool_timeout=0.1)
        try:
            registry.call_result(tool.name, {})
        finally:
            tool.release.set()
            self.wait_for_idle(registry)
        self.assertEqual(len(collector.results), 1)
        self.assertFalse(collector.results[0].success)
        self.assertEqual(collector.results[0].artifacts, ())

    def test_agent_continues_and_fuses_timed_out_tool_for_current_run(self):
        from scholaragent.agent import Agent
        from scholaragent.llm import ScriptedLLM

        tool = BlockingTool()
        registry = ToolRegistry([tool], tool_timeout=0.1)
        llm = ScriptedLLM([
            {"content": None, "tool_calls": [
                {"id": "first", "name": tool.name, "arguments": {}}]},
            {"content": None, "tool_calls": [
                {"id": "retry", "name": tool.name, "arguments": {}}]},
            {"content": "finished without retrying the side effect", "tool_calls": []},
        ])
        try:
            answer = Agent(llm, registry, verbose=False).run("test deadline")
            self.assertEqual(answer, "finished without retrying the side effect")
            self.assertEqual(tool.calls, 1)
            observations = [message for message in llm.last_messages
                            if message.get("role") == "tool"]
            self.assertEqual(len(observations), 2)
            self.assertTrue(all(message["content"].startswith(STOP_RETRY_PREFIX)
                                for message in observations))
        finally:
            tool.release.set()
            self.wait_for_idle(registry)

    def test_subset_preserves_timeout_and_shares_busy_state(self):
        tool = BlockingTool()
        registry = ToolRegistry([tool], tool_timeout=0.1)
        subset = registry.subset([tool.name])
        self.assertEqual(subset.tool_timeout, 0.1)
        self.assertIs(subset._execution, registry._execution)
        try:
            registry.call_result(tool.name, {})
            result = subset.call_result(tool.name, {})
            self.assertEqual(result.diagnostic["kind"], "tool_busy")
            self.assertTrue(result.stop_retry)
            self.assertEqual(tool.calls, 1)
        finally:
            tool.release.set()
            self.wait_for_idle(registry)

    def test_new_run_defers_reset_until_background_tool_completes(self):
        tool = BlockingTool()
        registry = ToolRegistry([tool], tool_timeout=0.1)
        registry.start_run()
        self.assertEqual(tool.resets, 1)
        try:
            registry.call_result(tool.name, {})
            registry.start_run()
            self.assertEqual(tool.resets, 1)
            self.assertEqual(registry.call_result(tool.name, {}).diagnostic["kind"],
                             "tool_busy")
        finally:
            tool.release.set()
            self.wait_for_idle(registry)
        self.assertTrue(registry.call_result(tool.name, {}).success)
        self.assertEqual(tool.resets, 2)
        self.wait_for_idle(registry)

    def test_saturation_rejects_instead_of_queuing_late_side_effects(self):
        tools = [BlockingTool(f"blocking_{i}") for i in range(4)]
        quick = QuickTool()
        registry = ToolRegistry([*tools, quick], tool_timeout=0.1)
        try:
            for tool in tools:
                self.assertEqual(registry.call_result(tool.name, {}).diagnostic["kind"],
                                 "timeout")
            result = registry.subset(["quick"]).call_result("quick", {})
            self.assertEqual(result.diagnostic["kind"], "executor_busy")
            self.assertTrue(result.stop_retry)
            self.assertEqual(quick.calls, 0)
            self.assertEqual(len(registry._execution.running), 4)
        finally:
            for tool in tools:
                tool.release.set()
            self.wait_for_idle(registry)
        self.assertEqual(quick.calls, 0)
        self.assertEqual(registry.call("quick", {}), "ok")
        self.wait_for_idle(registry)

    def test_tool_own_timeout_is_not_a_registry_deadline(self):
        class InternalTimeout(Tool):
            name = "internal_timeout"

            def run(self):
                raise TimeoutError("upstream socket timeout")

        registry = ToolRegistry([InternalTimeout()], tool_timeout=1)
        result = registry.call_result("internal_timeout", {})
        self.assertFalse(result.success)
        self.assertFalse(result.stop_retry)
        self.assertEqual(result.diagnostic["kind"], "exception")
        self.assertEqual(result.diagnostic["exception"], "TimeoutError")
        self.assertIn("upstream socket timeout", result.text)
        self.wait_for_idle(registry)

    def test_normal_exceptions_keep_existing_diagnostics(self):
        class Broken(Tool):
            name = "broken"

            def run(self):
                raise ValueError("bad input")

        registry = ToolRegistry([Broken()], tool_timeout=1)
        result = registry.call_result("broken", {})
        self.assertEqual(result.diagnostic["exception"], "ValueError")
        self.assertFalse(result.stop_retry)
        self.wait_for_idle(registry)


if __name__ == "__main__":
    unittest.main()
