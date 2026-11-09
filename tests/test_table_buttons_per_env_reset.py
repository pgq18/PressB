"""Per-slot seed compatibility for the existing single-world table task."""
import pytest

from pressb.tasks.table_buttons.backend import TableButtonsBackend
from pressb.tasks.table_buttons.task import COLORS
from test_table_buttons_backend import LAYOUTS, TelemetryEnvironment


def backend():
    return TableButtonsBackend({}, environment_factory=TelemetryEnvironment,
        layout_sampler=lambda profile,seed:{"positions":dict(zip(COLORS,LAYOUTS[profile]))})


@pytest.mark.parametrize("key",[0,"0"])
def test_per_environment_seed_reproduces_existing_singleton_spec(key):
    first,second=backend(),backend()
    options={"profile":"near","target_color":"blue"}
    expected=first.reset([0],123,options)
    actual=second.reset([0],999,options,env_seeds={key:123})
    assert actual==expected
    assert second.environment.spec==first.environment.spec
    metadata=second.describe()
    assert "per_env_reset_seeds" in metadata["capabilities"]
    assert metadata["num_envs"]==1 and metadata["execution"]["vectorized"] is False


@pytest.mark.parametrize("seeds",[{}, {1:123}, {0:1,1:2}, {0:1,"0":2}, {0:True}, {0:-1}])
def test_invalid_per_environment_seed_does_not_create_or_fault_environment(seeds):
    node=backend()
    with pytest.raises(ValueError):
        node.reset([0],12,env_seeds=seeds)
    assert node.environment is None and node._fault is None
