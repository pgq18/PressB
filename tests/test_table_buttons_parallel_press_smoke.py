"""CPU checks for the randomized-contact regression runner."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/smoke_table_buttons_parallel_press.py"
spec = importlib.util.spec_from_file_location("table_parallel_press_smoke",SCRIPT)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def snapshot():
    centers = np.array([[.3,.1,.8],[.4,.0,.8],[.3,-.1,.8]])
    positions = np.repeat(centers[:,None,:],3,axis=1)
    positions[:,1:,2] -= [.02,.01]
    return dict(button_body_positions=positions.reshape(-1,3).tolist(),
        button_spring_anchors=centers.tolist(),button_rest_world=centers.tolist())


def test_physical_axial_cap_travel_is_allowed_but_fixture_and_lateral_drift_are_detected():
    initial = snapshot()
    pressed = deepcopy(initial)
    pressed["button_body_positions"][0][2] -= .004
    assert max(smoke.fixture_errors(initial,pressed).values()) == 0.
    pressed["button_body_positions"][1][0] += .02
    assert smoke.fixture_errors(initial,pressed)["fixture_drift_m"] == pytest.approx(.02)
    pressed["button_spring_anchors"][0][0] += .01
    assert smoke.fixture_errors(initial,pressed)["anchor_rest_error_m"] == pytest.approx(.01)
    pressed["button_body_positions"][0][1] += .03
    assert smoke.fixture_errors(initial,pressed)["cap_lateral_error_m"] == pytest.approx(.03)


def test_plan_only_never_constructs_simulator(tmp_path):
    args = smoke.parser().parse_args(["--output",str(tmp_path/"result"),"--plan-only"])
    plan = dict(times=[0.,7.8], commands=[[0.]*7]*2)
    def forbidden(config):
        raise AssertionError("CPU preflight opened Isaac")
    assert smoke.run(args,plan_factory=lambda cfg,args:[[plan,plan]],backend_factory=forbidden) == 0
    report = json.loads((args.output/"report.json").read_text())
    assert report["status"] == "planned" and report["physical_execution"] is False
    assert report["planned_episodes"] == 2


def test_help_never_imports_simulator_or_task_modules():
    code = '''
import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('pressb','isaacsim','omni','carb','pxr'):
            raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
sys.argv=[sys.argv[1],'--help']
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    result = subprocess.run([sys.executable,"-c",code,str(SCRIPT)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert "--plan-only" in result.stdout


def test_two_round_runner_reuses_observer_and_resets_individual_slots(tmp_path):
    """Exercise runner lifetime across rounds; real PhysX is the separate smoke."""
    from pressb.runtime.contract import encode_image
    from pressb.runtime.parallel import SharedWorldBackend
    class Backend(SharedWorldBackend):
        def __init__(self, config):
            self.config = {**config,"scene":{"success":{"press_depth_m":.003}}}
            self.num_envs, self.ticks, self.timings = 2,np.zeros(2,dtype=int),{}
            self.execution = dict(physics_dt=1/120,physics_ticks_per_action=4,control_hz=30)
            self._observers, self.reset_calls = [],[]
            self.state = {str(i):snapshot() for i in range(2)}
            self.targets = {}
            self.closed = False
        def describe(self):
            return {}
        def add_observer(self, observer):
            self._observers.append(observer)
        def snapshot(self, ids=None):
            return {str(i):deepcopy(self.state[str(i)]) for i in ([0,1] if ids is None else ids)}
        def row(self, i):
            image = encode_image(np.zeros((2,3,3),np.uint8))
            return dict(env_id=i,observation=dict(state=[0.]*7,images={"up":image,"wrist":image}))
        def reset(self, ids, seed, options):
            self.reset_calls.append(ids)
            self.ticks[ids] = 0
            for i in ids:
                self.targets[i] = options["by_env"][str(i)]["target_color"]
            return [self.row(i) for i in ids]
        def step(self, commands):
            ids = [x["env_id"] for x in commands]
            self.ticks[ids] += 4
            outcomes = {i:dict(info={"physical_outcome":dict(travel_m={c:.004 for c in smoke.COLORS},
                finger_contact_force_n={c:.2 for c in smoke.COLORS},success=True)}) for i in ids}
            for observer in self._observers:
                observer.after_tick(self,ids,outcomes)
            return [{**self.row(i),"terminated":True,"truncated":False,"info":{"success":True}} for i in ids]
        def close(self):
            self.closed = True
    instances = []
    def factory(config):
        instances.append(Backend(config))
        return instances[-1]
    plan = dict(times=[0.,7.8],commands=[[0.]*7]*2,profile="medium",target_color="red",
        layout={"positions":{"red":[.3,.1],"yellow":[.4,0.],"blue":[.3,-.1]}},plan_report={})
    args = smoke.parser().parse_args(["--output",str(tmp_path/"result")])
    assert smoke.run(args,plan_factory=lambda cfg,args:[[deepcopy(plan) for i in range(2)] for r in range(2)],
                     backend_factory=factory) == 0
    assert instances[0].reset_calls == [[0,1],[0],[1]]
    assert len(instances[0]._observers) == 1 and instances[0].closed
    report = json.loads((args.output/"report.json").read_text())
    assert report["status"] == "passed" and len(report["rounds"]) == 2
    assert all(item["physics_samples"]==2 for item in report["rounds"])
