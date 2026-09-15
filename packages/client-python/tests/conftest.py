import pytest

import sbp.blackboard as blackboard_module


@pytest.fixture(autouse=True)
def reset_shared_blackboard():
    """local=True always routes through one process-wide singleton (get_shared_blackboard) --
    reset it before each test so agent-level tests don't leak state into each other."""
    blackboard_module._shared_blackboard = None
    yield
    blackboard_module._shared_blackboard = None
