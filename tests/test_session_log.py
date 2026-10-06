"""Faithful per-session transcripts, including text removed by context trimming."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from nanoclaw.core.session_log import SessionTranscript, search_session_transcript_tool


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


class SessionTranscriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = SessionTranscript(self.tmp.name)

    def test_file_is_bound_to_one_session_and_keeps_full_text(self):
        long_text = "detail-" + ("n" * 500)
        messages = [
            SystemMessage(content="ephemeral prompt", id="sys"),
            HumanMessage(content="记得这个数字 42", id="h1"),
            AIMessage(content="", id="a1", tool_calls=[{
                "name": "lookup",
                "args": {"help_token": "secret-token", "q": long_text},
                "id": "call-1",
                "type": "tool_call",
            }]),
            ToolMessage(content=f"help_token: secret-token\n{long_text}", name="lookup",
                        tool_call_id="call-1", id="t1"),
        ]
        self.assertEqual(self.log.append_messages("sess-a", "thread-1", messages), 3)

        path = self.log.path_for("sess-a")
        self.assertTrue(path.endswith(os.path.join("sess-a.jsonl")))
        rows = _lines(path)
        self.assertEqual(rows[0]["event"], "session_start")
        self.assertEqual(rows[0]["session_id"], "sess-a")
        self.assertEqual(rows[0]["thread_id"], "thread-1")
        self.assertTrue(all(row["session_id"] == "sess-a" for row in rows))
        stored = {row["message_id"]: row for row in rows if row["event"] == "message"}
        self.assertNotIn("sys", stored)
        self.assertEqual(stored["h1"]["content"], "记得这个数字 42")
        self.assertEqual(stored["a1"]["tool_calls"][0]["args"]["help_token"], "[REDACTED]")
        self.assertIn(long_text, stored["a1"]["tool_calls"][0]["args"]["q"])
        self.assertIn(long_text, stored["t1"]["content"])
        self.assertNotIn("secret-token", stored["t1"]["content"])
        self.assertEqual(stored["t1"]["tool_call_id"], "call-1")

        self.assertEqual(self.log.append_messages("sess-a", "thread-1", messages), 0)
        self.assertEqual(len(_lines(path)), 4)

    def test_new_session_does_not_copy_messages_already_stored(self):
        self.log.append_messages("sess-a", "thread-1", [HumanMessage(content="old", id="h1")])
        fresh = SessionTranscript(self.tmp.name)
        written = fresh.append_messages("sess-b", "thread-1", [
            HumanMessage(content="old", id="h1"),
            HumanMessage(content="new", id="h2"),
        ])
        self.assertEqual(written, 1)
        old_rows = _lines(fresh.path_for("sess-a"))
        new_rows = _lines(fresh.path_for("sess-b"))
        self.assertEqual([row.get("message_id") for row in old_rows if row["event"] == "message"], ["h1"])
        self.assertEqual([row.get("message_id") for row in new_rows if row["event"] == "message"], ["h2"])
        self.assertTrue(all(row["session_id"] == "sess-b" for row in new_rows))

    def test_session_rejects_a_different_thread(self):
        self.log.append_messages("sess-a", "thread-1", [HumanMessage(content="hi", id="h1")])
        with self.assertRaisesRegex(ValueError, "已绑定"):
            self.log.append_messages("sess-a", "thread-2", [HumanMessage(content="other", id="h2")])
        self.assertEqual([row.get("message_id") for row in _lines(self.log.path_for("sess-a"))
                          if row["event"] == "message"], ["h1"])

    def test_session_id_must_be_a_safe_filename(self):
        with self.assertRaisesRegex(ValueError, "session_id"):
            self.log.append_messages("../other", "thread-1", [HumanMessage(content="x", id="h1")])
        self.assertEqual(list(Path(self.tmp.name).glob("*.jsonl")), [])

    def test_search_returns_neighboring_detail_for_this_thread_only(self):
        self.log.append_messages("sess-a", "thread-1", [
            HumanMessage(content="服务端口当时改成了 8841", id="h1"),
            AIMessage(content="已按 8841 继续配置", id="a1"),
        ])
        self.log.append_messages("sess-b", "thread-2", [
            HumanMessage(content="另一条线程的端口是 8841", id="other"),
        ])
        found = self.log.search_thread("thread-1", "8841")
        self.assertIn("8841", found)
        self.assertIn("已按 8841 继续配置", found)
        self.assertIn("message_id=a1", found)
        self.assertNotIn("另一条线程", found)
        self.assertNotIn("message_id=other", found)
        schema = search_session_transcript_tool.tool_call_schema.model_json_schema()["properties"]
        self.assertEqual(set(schema), {"query"})

        with patch("nanoclaw.core.session_log.session_transcript", self.log):
            scoped = search_session_transcript_tool.invoke(
                {"query": "8841"},
                config={"configurable": {"transcript_thread_id": "thread-1"}},
            )
        self.assertIn("message_id=h1", scoped)
        self.assertNotIn("另一条线程", scoped)


class AgentSessionLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = SessionTranscript(self.tmp.name)
        self.addCleanup(patch.stopall)

    def _app(self, model):
        patch("nanoclaw.core.agent.get_provider", return_value=model).start()
        patch("nanoclaw.core.agent.session_transcript", self.log).start()
        from nanoclaw.core.agent import create_agent_app
        return create_agent_app(tools=[])

    def test_without_session_id_nothing_is_written(self):
        class Model:
            def bind_tools(self, tools):
                return self
            def invoke(self, messages, config=None):
                return AIMessage(content="hi", id="a1")

        app = self._app(Model())
        app.invoke(
            {"messages": [HumanMessage(content="hello", id="h1")], "summary": ""},
            config={"configurable": {"thread_id": "thread-1"}},
        )
        self.assertEqual(list(Path(self.tmp.name).glob("*.jsonl")), [])

    def test_trimmed_turns_remain_in_the_session_file(self):
        class Model:
            def bind_tools(self, tools):
                return self
            def invoke(self, messages, config=None):
                first = messages[0]
                if first.type == "human" and "交接文档" in str(first.content):
                    return AIMessage(content="摘要", id="summary-1")
                return AIMessage(content="继续", id="reply-1")

        history = []
        for i in range(40):
            history.append(HumanMessage(content=f"user-{i}-" + ("x" * 80), id=f"h{i}"))
            history.append(AIMessage(content=f"ai-{i}", id=f"a{i}"))
        patch("nanoclaw.core.agent.print_formatted_text").start()
        app = self._app(Model())
        result = app.invoke(
            {"messages": history, "summary": ""},
            config={"configurable": {"thread_id": "thread-1", "session_id": "sess-trim"}},
        )
        kept = " ".join(str(m.content) for m in result["messages"])
        self.assertNotIn("user-0-", kept)
        rows = _lines(self.log.path_for("sess-trim"))
        stored = {row["message_id"]: row for row in rows if row["event"] == "message"}
        self.assertEqual(stored["h0"]["content"], history[0].content)
        self.assertEqual(stored["a29"]["content"], "ai-29")
        self.assertEqual(stored["reply-1"]["content"], "继续")
        self.assertTrue(all(row["session_id"] == "sess-trim" for row in rows))

    def test_tool_result_is_stored_in_full_on_the_same_session(self):
        payload = "payload-" + ("z" * 400)

        @tool
        def echo_long(text: str) -> str:
            """Return the given text unchanged."""
            return text

        class Model:
            def bind_tools(self, tools):
                return self
            def invoke(self, messages, config=None):
                if not any(m.type == "tool" for m in messages):
                    return AIMessage(content="", id="ai-call", tool_calls=[{
                        "name": "echo_long",
                        "args": {"text": payload},
                        "id": "call-1",
                        "type": "tool_call",
                    }])
                return AIMessage(content="done", id="ai-final")

        patch("nanoclaw.core.agent.get_provider", return_value=Model()).start()
        patch("nanoclaw.core.agent.session_transcript", self.log).start()
        from nanoclaw.core.agent import create_agent_app
        app = create_agent_app(tools=[echo_long])
        app.invoke(
            {"messages": [HumanMessage(content="read it", id="h1")], "summary": ""},
            config={"configurable": {"thread_id": "thread-1", "session_id": "sess-tool"}},
        )
        stored = {row["message_id"]: row for row in _lines(self.log.path_for("sess-tool"))
                  if row["event"] == "message"}
        tool_rows = [row for row in stored.values() if row["type"] == "tool"]
        self.assertEqual(len(tool_rows), 1)
        self.assertEqual(tool_rows[0]["content"], payload)
        self.assertEqual(tool_rows[0]["session_id"], "sess-tool")
        self.assertEqual(stored["ai-final"]["content"], "done")

    def test_summary_asks_the_main_agent_to_delegate_missing_detail(self):
        seen = {}

        class Model:
            def bind_tools(self, tools):
                return self
            def invoke(self, messages, config=None):
                seen["prompt"] = messages[0].content
                return AIMessage(content="ok", id="a-reply")

        patch("nanoclaw.core.agent.get_provider", return_value=Model()).start()
        patch("nanoclaw.core.agent.session_transcript", self.log).start()
        from nanoclaw.core.agent import create_agent_app
        from nanoclaw.core.subagents import default_subagents
        app = create_agent_app(tools=[], enable_subagents=True)
        app.invoke(
            {"messages": [HumanMessage(content="端口是多少", id="h-ask")], "summary": "之前改过端口"},
            config={"configurable": {"thread_id": "thread-1", "session_id": "sess-ask"}},
        )
        self.assertIn("之前改过端口", seen["prompt"])
        self.assertIn("session_researcher", seen["prompt"])

        plain = create_agent_app(
            tools=[], enable_subagents=True,
            subagent_specs=tuple(role for role in default_subagents() if role.name != "session_researcher"))
        plain.invoke(
            {"messages": [HumanMessage(content="端口是多少", id="h-plain")], "summary": "之前改过端口"},
            config={"configurable": {"thread_id": "thread-1", "session_id": "sess-plain"}},
        )
        self.assertNotIn("session_researcher", seen["prompt"])


if __name__ == "__main__":
    unittest.main()
