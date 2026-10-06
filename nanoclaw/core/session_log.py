"""Append-only transcript bound to one session id.

This is the recoverable conversation text. It is separate from the audit log:
writes are synchronous, message bodies are not truncated, and each file belongs
to exactly one session.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from .config import SESSIONS_DIR

_HELP_TOKEN_TEXT = re.compile(r"help_token: [A-Za-z0-9_-]+")


def _safe_session_id(session_id: str) -> str | None:
    if not isinstance(session_id, str) or not session_id:
        return None
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in session_id):
        return None
    return session_id


def _redact(value):
    if isinstance(value, dict):
        return {k: "[REDACTED]" if k == "help_token" else _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if isinstance(value, str):
        return _HELP_TOKEN_TEXT.sub("help_token: [REDACTED]", value)
    return value


def _jsonable(value):
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))


class SessionTranscript:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self._lock = threading.Lock()
        self._seen: set[str] | None = None
        self._owners: dict[str, str] = {}

    def path_for(self, session_id: str) -> str | None:
        safe = _safe_session_id(session_id)
        if not safe:
            return None
        return os.path.join(self.root_dir, safe + ".jsonl")

    def append_messages(self, session_id: str, thread_id: str, messages) -> int:
        """Append conversation messages that have not been stored in any session file.

        Returns the number of newly written messages. A session file is created
        on the first write and keeps the thread id from that write.
        """
        safe = _safe_session_id(session_id)
        if not safe:
            raise ValueError("session_id 只能包含字母、数字、下划线和连字符")
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("记录会话日志需要非空 thread_id")

        records = []
        for message in messages:
            if isinstance(message, SystemMessage) or getattr(message, "type", None) == "system":
                continue
            if not isinstance(message, BaseMessage):
                continue
            message_id = message.id
            if not message_id:
                continue
            records.append(self._message_record(safe, thread_id, message_id, message))

        with self._lock:
            if self._seen is None:
                self._seen = self._scan_seen()
            fresh = []
            batch_ids = set()
            for record in records:
                message_id = record["message_id"]
                if message_id in self._seen or message_id in batch_ids:
                    continue
                batch_ids.add(message_id)
                fresh.append(record)
            if not fresh:
                return 0
            path = os.path.join(self.root_dir, safe + ".jsonl")
            os.makedirs(self.root_dir, exist_ok=True)
            owner = self._owner(safe, path)
            if owner is not None and owner != thread_id:
                raise ValueError(f"会话 {safe} 已绑定 thread_id={owner}")
            now = datetime.now(timezone.utc).isoformat()
            with open(path, "a", encoding="utf-8") as stream:
                if owner is None:
                    header = {
                        "event": "session_start",
                        "ts": now,
                        "session_id": safe,
                        "thread_id": thread_id,
                    }
                    stream.write(json.dumps(header, ensure_ascii=False) + "\n")
                    self._owners[safe] = thread_id
                for record in fresh:
                    record["ts"] = now
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._seen.update(record["message_id"] for record in fresh)
            return len(fresh)

    def _message_record(self, session_id: str, thread_id: str, message_id: str, message: BaseMessage) -> dict:
        tool_calls = getattr(message, "tool_calls", None) or None
        return {
            "event": "message",
            "ts": "",
            "session_id": session_id,
            "thread_id": thread_id,
            "message_id": message_id,
            "type": message.type,
            "name": getattr(message, "name", None),
            "content": _jsonable(_redact(message.content)),
            "tool_calls": _jsonable(_redact(tool_calls)) if tool_calls else None,
            "tool_call_id": getattr(message, "tool_call_id", None),
        }

    def _owner(self, session_id: str, path: str) -> str | None:
        if session_id in self._owners:
            return self._owners[session_id]
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return None
        with open(path, "r", encoding="utf-8") as stream:
            line = stream.readline()
        try:
            header = json.loads(line)
        except json.JSONDecodeError:
            header = {}
        owner = header.get("thread_id") if header.get("event") == "session_start" else None
        if isinstance(owner, str) and owner:
            self._owners[session_id] = owner
            return owner
        return None

    def _scan_seen(self) -> set[str]:
        seen: set[str] = set()
        if not os.path.isdir(self.root_dir):
            return seen
        for name in os.listdir(self.root_dir):
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(self.root_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if item.get("event") == "message" and item.get("message_id"):
                            seen.add(item["message_id"])
            except OSError:
                continue
        return seen

    def search_thread(self, thread_id: str, query: str, *, limit: int = 6, neighbors: int = 1,
                      max_chars: int = 8000) -> str:
        """Return transcript excerpts for one thread whose text contains any query term.

        Matches keep one neighboring message on each side so a short answer next
        to the matched line is included. The caller cannot choose a file path.
        """
        if not isinstance(thread_id, str) or not thread_id.strip():
            return "没有可检索的会话线程。"
        terms = [term.casefold() for term in str(query).split() if term.strip()]
        if not terms:
            return "query 不能为空。"
        sessions = self._sessions_for_thread(thread_id.strip())
        if not sessions:
            return "当前线程没有已落盘的会话原文。"

        hits = []
        order = 0
        for session_id, messages in sessions:
            for index, record in enumerate(messages):
                haystack = _search_text(record).casefold()
                score = sum(1 for term in terms if term in haystack)
                if score:
                    hits.append((score, order, session_id, messages, index))
                order += 1
        if not hits:
            return "没有找到与这些关键词匹配的会话原文。"

        hits.sort(key=lambda item: (-item[0], item[1]))
        chosen = {}
        for _, _, session_id, messages, index in hits[:limit]:
            start = max(0, index - neighbors)
            end = min(len(messages), index + neighbors + 1)
            span = chosen.setdefault(session_id, {"messages": messages, "ranges": []})
            span["ranges"].append((start, end))

        lines = []
        used = 0
        for session_id, messages in sessions:
            span = chosen.get(session_id)
            if not span:
                continue
            for start, end in _merge_ranges(span["ranges"]):
                for record in messages[start:end]:
                    block = _format_record(session_id, record)
                    if used and used + len(block) > max_chars:
                        lines.append("...[匹配结果过长，已截断]")
                        return "\n\n".join(lines)
                    lines.append(block)
                    used += len(block)
        return "\n\n".join(lines)

    def _sessions_for_thread(self, thread_id: str) -> list[tuple[str, list[dict]]]:
        root = Path(self.root_dir).resolve()
        if not root.is_dir():
            return []
        loaded = []
        for path in root.glob("*.jsonl"):
            resolved = path.resolve()
            if not resolved.is_file() or not resolved.is_relative_to(root):
                continue
            session_id = path.stem
            started = ""
            messages = []
            try:
                with open(resolved, "r", encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if item.get("event") == "session_start" and not started:
                            started = str(item.get("ts") or "")
                        elif item.get("event") == "message" and item.get("thread_id") == thread_id:
                            messages.append(item)
            except OSError:
                continue
            if messages:
                loaded.append((started, session_id, messages))
        loaded.sort(key=lambda item: (item[0], item[1]))
        return [(session_id, messages) for _, session_id, messages in loaded]


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _search_text(record: dict) -> str:
    parts = [_content_text(record.get("content")), str(record.get("name") or "")]
    calls = record.get("tool_calls") or []
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, dict):
                parts.append(str(call.get("name") or ""))
                args = call.get("args")
                parts.append(json.dumps(args, ensure_ascii=False, default=str) if isinstance(args, (dict, list)) else str(args or ""))
    return "\n".join(parts)


def _format_record(session_id: str, record: dict) -> str:
    content = _content_text(record.get("content"))
    if len(content) > 1200:
        content = content[:1200] + "\n...[单条过长，已截断]"
    header = (
        f"session_id={session_id} message_id={record.get('message_id')} "
        f"type={record.get('type')} name={record.get('name') or '-'}"
    )
    return header + "\n" + content


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


class _TranscriptQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500, description="要找回的细节关键词。用原文里可能出现的短词，多个词用空格分开；不要传文件路径。")


def search_session_transcript(query: str, config: RunnableConfig = None) -> str:
    """Search the current thread's saved transcript. The thread id comes from the host, not the model."""
    thread_id = ((config or {}).get("configurable") or {}).get("transcript_thread_id")
    if not isinstance(thread_id, str) or not thread_id.strip():
        return "没有可检索的会话线程。"
    return session_transcript.search_thread(thread_id, query)


search_session_transcript_tool = StructuredTool.from_function(
    func=search_session_transcript,
    name="search_session_transcript",
    args_schema=_TranscriptQuery,
    description=(
        "检索当前对话线程已经落盘的会话原文，用来补回摘要里缺少的细节。"
        "只接受关键词 query，不能指定文件路径，也不能读取其他线程。"
        "返回匹配消息及其相邻一条原文，并标出 session_id 与 message_id。"
    ),
)


session_transcript = SessionTranscript(SESSIONS_DIR)
