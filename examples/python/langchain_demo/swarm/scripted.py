"""Deterministic fake models and a tiny fixed corpus for --mode scripted: zero API
cost, exact tool-call sequences instead of live reasoning. Follows rogue_demo.py's
ScriptedChatModel pattern (FakeMessagesListChatModel + a bind_tools override, since
create_agent() always calls bind_tools and the base class raises on it).
"""
from __future__ import annotations

import uuid
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class ScriptedChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": uuid.uuid4().hex}])


def _done(text: str = "done") -> AIMessage:
    return AIMessage(content=text)


SCRIPTED_FILES = {
    "app/auth/LoginActivity.java": (
        'public class LoginActivity {\n'
        '    private static final String API_KEY = "sk-hardcoded-1234567890";\n'
        '    // ... login logic ...\n}\n'
    ),
    "app/auth/SessionManager.java": (
        'public class SessionManager {\n'
        '    private static final String SECRET = "hardcoded-session-secret";\n'
        '    // ... session logic ...\n}\n'
    ),
    "app/util/Logger.java": (
        'public class Logger {\n'
        '    public static void log(String tag, String msg) {\n'
        '        Log.d(tag, "password=" + msg);  // logs sensitive values\n    }\n}\n'
    ),
    "app/util/CryptoHelper.java": (
        'public class CryptoHelper {\n'
        '    public static byte[] hash(byte[] data) {\n'
        '        return MessageDigest.getInstance("SHA-256").digest(data);\n    }\n}\n'
    ),
}


def write_scripted_corpus(root: Path) -> None:
    for rel, content in SCRIPTED_FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def scout_model(file_plan: list[tuple[str, str | None]]) -> ScriptedChatModel:
    """file_plan: [(path, kind_or_None), ...] -- the files this scout will process, in
    order, and the evidence kind it will report for each (None = nothing notable). The
    last file's activation omits continue_watching, ending the scout's own loop."""
    messages: list[AIMessage] = []
    for i, (path, kind) in enumerate(file_plan):
        is_last = i == len(file_plan) - 1
        messages += [
            _call("claim_file", path=path),
            _call("read_file", path=path),
        ]
        if kind:
            messages.append(_call("report_evidence", path=path, kind=kind))
        messages += [
            _call("mark_visited", path=path),
            _call("mark_covered", path=path),
        ]
        if not is_last:
            messages.append(_call("continue_watching"))
        messages.append(_done())
    return ScriptedChatModel(responses=messages)


def bloodhound_model() -> ScriptedChatModel:
    """One real activation confirming app/auth (2 independent hardcoded_secret
    reports) while leaving app/util unconfirmed (Logger alone), plus spare no-op
    activations in case cooldown splits the evidence emissions across more than one
    trigger -- FakeMessagesListChatModel cycles back to message 0 once exhausted, so
    an unplanned activation replays this real one rather than erroring, which is safe
    because every emit here uses merge_strategy="reinforce"."""
    real = [
        _call("read_file", path="app/auth/LoginActivity.java"),
        _call("read_file", path="app/auth/SessionManager.java"),
        _call("sbp_inscribe", trail="dossier", key="app/auth", value={
            "files": ["app/auth/LoginActivity.java", "app/auth/SessionManager.java"],
            "kind": "hardcoded_secret",
            "summary": "Hardcoded secrets in two independent files under app/auth.",
        }),
        _call("mark_hot", dir_path="app/auth", kind="hardcoded_secret", intensity=1.0),
        _call("ask_question", topic="check_util_logging"),
        _done(),
    ]
    return ScriptedChatModel(responses=real)


def judge_model() -> ScriptedChatModel:
    messages = [
        _call("sbp_read", trails=["dossier"]),
        _call("sbp_sniff", trails=["swarm.hot"]),
        _call("sbp_read", trails=["ledger"]),
        _call("write_report", markdown=(
            "# Foraging swarm report\n\n"
            "## Confirmed\n"
            "- **app/auth**: hardcoded_secret, confirmed independently in "
            "LoginActivity.java and SessionManager.java. See dossier `app/auth`.\n\n"
            "## Hypotheses\n"
            "- **app/util**: possible insecure logging in Logger.java (single report, "
            "not independently confirmed -- not promoted to confirmed).\n"
        )),
        _done(),
    ]
    return ScriptedChatModel(responses=messages)
