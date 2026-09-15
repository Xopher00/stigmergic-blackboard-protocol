import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

import sbp.blackboard as blackboard_module


@pytest.fixture(autouse=True)
def reset_shared_blackboard():
    """Same reasoning as packages/client-python/tests/conftest.py -- local=True routes
    through one process-wide singleton, reset it so tests don't leak into each other."""
    blackboard_module._shared_blackboard = None
    yield
    blackboard_module._shared_blackboard = None


class ScriptedChatModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel plays back a scripted list of AIMessages (with or
    without tool_calls) in order -- exactly what's needed to drive create_agent()
    through a controlled ReAct loop with zero real API calls. It doesn't support
    bind_tools out of the box (raises NotImplementedError); create_agent() always
    calls it, so this override just returns self, ignoring the tool schema -- the
    scripted responses already specify which tool to call, so there's nothing for
    real schema validation to do here."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@pytest.fixture
def make_model():
    def _make(responses):
        return ScriptedChatModel(responses=responses)
    return _make
