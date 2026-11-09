"""Table-button task hooks; shared runtime owns clocks, rendering and scheduling."""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from pressb.robots.piper.kinematics import PiperKinematics
from pressb.runtime.parallel_contract import ParallelTask, SceneTemplate
from .backend import ROOT, STATE_NAMES, STATE_UNITS, load_config, sample_reset_spec, task_identity
from .collision_policy import is_allowed_finger_closure, strict_outcome
from .task import COLORS, TableButtonOutcome

EXECUTION_KEYS = ("num_envs", "gpu", "horizon", "physics_dt", "control_hz", "max_episode_seconds",
    "settling_ticks", "gamma", "headless", "spacing_m", "render_resolution", "renderer_profile",
    "render_subframes", "observation_render_frames", "reset_render_frames", "cpu_threads",
    "png_workers", "log_dir", "image_preprocessing", "image_output_resolution")


class TableButtonsTask(ParallelTask):
    def __init__(self, config, *, layout_sampler=None):
        self.config = load_config(config)
        self.execution = {key: deepcopy(self.config[key]) for key in EXECUTION_KEYS if key in self.config}
        self.execution.update(terminal_boundary="control_interval", timeout_reason="timeout")
        self.execution.setdefault("log_dir", str(ROOT / "outputs/table_buttons_parallel"))
        self.kin = PiperKinematics(self.config["scene"]["robot_urdf"], tip_offset=.1358)
        self.lower = np.r_[self.kin.lower, 0.]
        self.upper = np.r_[self.kin.upper, self.config["scene"]["gripper_max_width_m"]]
        self.identity = task_identity(self.config)
        self._layout_sampler, self._placements = layout_sampler, {}
        self.checkers, self.contact_forces, self.blocking_contacts = {}, {}, {}
        self.wrong_contact, self.first_wrong_press, self.last_outcome = {}, {}, {}
        # hpp-fcl/Pinocchio must load their Assimp dependency before Kit loads
        # its bundled version. Preparing collision geometry is CPU-only and
        # also avoids first-reset initialization in the simulation thread.
        if self.config["randomization"]["positions"] and layout_sampler is None:
            for profile in self.config["randomization"]["profiles"]:
                self._placement(profile)

    def metadata(self):
        cfg, width = self.config, self.config["action_dim"]
        result = dict(schema_version=2, node_type="simulation", task_id="table_buttons", num_envs=cfg["num_envs"],
            **self.identity,
            observation={"state": {"names": STATE_NAMES.copy(), "units": STATE_UNITS.copy()}, "cameras": ["up", "wrist"]},
            action={"representation": "joint_position", "frame": "robot_base", "names": STATE_NAMES[:width],
                "units": STATE_UNITS[:width], "horizon": cfg["horizon"], "control_hz": cfg["control_hz"],
                "lower": self.lower[:width].tolist(), "upper": self.upper[:width].tolist()},
            control_state_dim=7, task_features_dim=0, reset_mode="indexed",
            discount={"gamma": cfg["gamma"], "timebase_hz": cfg["control_hz"]}, smoothing=deepcopy(cfg["smoothing"]),
            reward_schema={"kind": "sparse_physical_press", "success": 1., "failure": 0., "timeout": 0.,
                "within_chunk_discount": "gamma ** event_control_interval_index"},
            success_rule="Measured cap displacement + finger force + dwell; strict contact veto through the control boundary",
            collision_policy=deepcopy(cfg["collision_policy"]), randomization=deepcopy(cfg["randomization"]))
        if width == 6:
            result["fixed_gripper_width_m"] = cfg["fixed_gripper_width_m"]
        return result

    def build_template(self):
        from pxr import UsdPhysics
        from pressb.robots.piper.isaac import build_piper_runtime
        from pressb.runtime.workbench import build_tripod
        from .scene import build_scene

        def author(stage, cfg):
            scene = build_scene(stage, cfg)
            # Preserve original collider geometry, paths and spring-to-world binding.
            # Two kinematic fixture bodies per button permit indexed tensor resets
            # without moving a static USD collider or teleporting only the cap.
            for button in scene["buttons"].values():
                parent = button["body_path"].rsplit("/", 1)[0]
                button["fixture_paths"] = [parent + "/Housing", parent + "/Bezel"]
                for path in button["fixture_paths"]:
                    body = UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(path))
                    body.CreateRigidBodyEnabledAttr(True)
                    body.CreateKinematicEnabledAttr(True)
            return scene

        runtime = build_piper_runtime(deepcopy(self.config["scene"]), author, asset_root=ROOT,
            camera_hardware_builder=build_tripod, initialize=False)
        return SceneTemplate(runtime.world, runtime.stage, runtime.robot_path, runtime.camera_paths,
            data={"scene": runtime.scene, "camera_specs": runtime.camera_specs})

    def create_robot(self, engine):
        from pressb.robots.piper.parallel import PiperJointDriver
        cfg = self.config
        initial = [*cfg["scene"]["initial_joint_positions_rad"], cfg["scene"]["initial_gripper_width_m"]]
        return PiperJointDriver(engine, cfg["scene"], initial, action_dim=cfg["action_dim"],
            fixed_gripper_width_m=cfg["fixed_gripper_width_m"], smoothing=cfg["smoothing"])

    def _sample(self, profile, seed):
        if self._layout_sampler is not None:
            return self._layout_sampler(profile, seed)
        return self._placement(profile).sample(seed)

    def _placement(self, profile):
        from .placement import ReachablePlacement
        if profile not in self._placements:
            cfg = deepcopy(self.config["scene"])
            cfg["position_sampling"]["spacing_profile"] = profile
            self._placements[profile] = ReachablePlacement(self.kin, cfg)
        return self._placements[profile]

    def sample_reset(self, env_id, seed, options):
        return sample_reset_spec(self, seed, options, env_id=env_id)

    def bind(self, engine):
        super().bind(engine)
        self.scene = engine.template.data["scene"]
        self.buttons = self.scene["buttons"]
        self.body_paths = []
        self.button_rows = {}
        for color in COLORS:
            button = self.buttons[color]
            parent = button["body_path"].rsplit("/", 1)[0]
            paths = [button["body_path"], *button.get("fixture_paths", [parent + "/Housing", parent + "/Bezel"])]
            self.button_rows[color] = np.arange(len(self.body_paths), len(self.body_paths) + 3)
            self.body_paths.extend(paths)
        self.bodies, self.rows = engine.rigid_bodies(self.body_paths)
        self.rest_world = np.array([[np.asarray(self.buttons[color]["cap_rest_world"]) + engine.offsets[eid]
            for color in COLORS] for eid in range(engine.num_envs)], dtype=np.float64)

    def after_settle(self):
        positions, orientations = self.bodies.get_world_poses()
        self.initial_positions = positions[self.rows].copy()
        self.initial_orientations = orientations[self.rows].copy()

    def _anchor_attribute(self, env_id, color):
        path = self.engine.relocate(env_id, self.buttons[color]["joint_path"])
        return self.engine.stage.GetPrimAtPath(path).GetAttribute("physics:localPos0")

    def reset_task(self, env_ids, specs):
        positions, orientations, indices, cap_indices, info = [], [], [], [], {}
        for eid in env_ids:
            spec = specs[eid]
            local = self.initial_positions[eid].copy()
            for color_index, color in enumerate(COLORS):
                button = self.buttons[color]
                delta = np.r_[np.subtract(spec["positions"][color], button["center_xy_base"]), 0.]
                local[self.button_rows[color]] += delta
                rest = np.asarray(button["cap_rest_world"], dtype=float) + self.engine.offsets[eid] + delta
                local[self.button_rows[color][0]] = rest
                self.rest_world[eid, color_index] = rest
                # A world-side anchor is in absolute world coordinates. Rewriting
                # only cap transforms would pull the cap back to its old XY.
                self._anchor_attribute(eid, color).Set(tuple(float(x) for x in rest))
            positions.extend(local)
            orientations.extend(self.initial_orientations[eid])
            indices.extend(self.rows[eid])
            cap_indices.extend(self.rows[eid, self.button_rows[color][0]] for color in COLORS)
            self.checkers[eid] = TableButtonOutcome(spec["target_color"], **self.config["scene"]["success"])
            self.contact_forces[eid] = dict.fromkeys(COLORS, 0.)
            self.blocking_contacts[eid] = []
            self.wrong_contact[eid], self.first_wrong_press[eid], self.last_outcome[eid] = False, None, {}
            info[eid] = {**deepcopy(spec), "button_teleport_after_reset": False,
                "fixture_reset": "cap_housing_bezel_and_world_spring_anchor", "zero_tick_reset": True}
        self.bodies.set_world_poses(np.asarray(positions), np.asarray(orientations), np.asarray(indices, dtype=np.int64))
        # PhysX disallows velocity setters for kinematic Housing/Bezel bodies.
        # Their transforms are reset above; only dynamic caps carry velocity.
        self.bodies.set_velocities(np.zeros((len(cap_indices), 6)), np.asarray(cap_indices, dtype=np.int64))
        return info

    def before_tick(self, env_ids):
        for eid in env_ids:
            self.contact_forces[eid] = dict.fromkeys(COLORS, 0.)

    def on_contact(self, env_id, actor_paths, contacts):
        if env_id not in self.checkers or not self.engine.backend.ready[env_id] or not contacts:
            return
        if len(actor_paths) != 4:
            raise ValueError("Table contact routing requires actor0/actor1/collider0/collider1")
        actors, colliders = list(actor_paths[:2]), list(actor_paths[2:])
        robot_path = self.engine.template.robot_path
        def robot_link(actor, collider):
            for path in (actor, collider):
                if path.startswith(robot_path + "/"):
                    return path[len(robot_path)+1:].split("/", 1)[0]
            return None
        links = [robot_link(actor, collider) for actor, collider in zip(actors, colliders)]
        fingers = any(link in ("link7", "link8") for link in links)
        caps = [color for color, button in self.buttons.items() if any(
            path == button["body_path"] or path.startswith(button["body_path"] + "/") for path in actor_paths)]
        force = sum(float(np.linalg.norm(np.asarray(contact.impulse, dtype=float))) for contact in contacts) / self.config["physics_dt"]
        if fingers:
            for color in caps:
                self.contact_forces[env_id][color] += force
        moving = any(link is not None and link != "base_link" for link in links)
        self_contact = all(link is not None for link in links) and links[0] != links[1]
        environment_contact = moving and any(link is None for link in links) and not caps
        non_finger_cap = bool(caps) and not fingers
        normalized_actors = [f"/World/Piper/{link}" if link is not None else actor for actor, link in zip(actors, links)]
        if force > .02 and moving and (self_contact or environment_contact or non_finger_cap):
            if not is_allowed_finger_closure(normalized_actors, self.config["collision_policy"]):
                self.blocking_contacts[env_id].append({"actors": actors, "colliders": colliders, "force_n": force})

    def after_tick(self, env_ids):
        positions, orientations = self.bodies.get_world_poses()
        velocities = self.bodies.get_velocities()
        if not all(np.isfinite(value).all() for value in (positions, orientations, velocities)):
            raise RuntimeError("Nonfinite table button physics state")
        results = {}
        for eid in env_ids:
            spec = self.engine.backend.reset_specs[eid]
            seconds = float(self.engine.backend.ticks[eid]) * self.config["physics_dt"]
            travels = {color: float(self.rest_world[eid, j, 2] - positions[self.rows[eid, self.button_rows[color][0]], 2])
                for j, color in enumerate(COLORS)}
            force = self.contact_forces[eid]
            self.wrong_contact[eid] |= any(force[color] > .02 for color in COLORS if color != spec["target_color"])
            result = self.checkers[eid].update(seconds, travels, force)
            wrong = [color for color in result["press_events"] if color != spec["target_color"]]
            if wrong and self.first_wrong_press[eid] is None:
                self.first_wrong_press[eid] = {"color": wrong[0], "time_s": seconds}
            success, reason = strict_outcome(result, wrong_press=self.first_wrong_press[eid],
                wrong_contact=self.wrong_contact[eid], unexpected_contacts=self.blocking_contacts[eid])
            terminated = bool(result["terminated"] or self.wrong_contact[eid] or self.blocking_contacts[eid])
            self.last_outcome[eid] = deepcopy(result)
            results[eid] = dict(reward=float(result["reward"] if success else 0.), success=bool(success),
                terminated=terminated, reason=reason if terminated else None,
                info={"physical_outcome": result, "wrong_contact": bool(self.wrong_contact[eid]),
                    "first_wrong_press": deepcopy(self.first_wrong_press[eid]),
                    "blocking_contact_count": len(self.blocking_contacts[eid]), "target_color": spec["target_color"],
                    "spacing_profile": spec["profile"], "button_teleport_after_reset": False})
        return results

    def diagnostic_state(self, env_ids):
        positions, orientations = self.bodies.get_world_poses()
        velocities = self.bodies.get_velocities()
        return {eid: {"button_body_positions": positions[self.rows[eid]].tolist(),
            "button_body_orientations": orientations[self.rows[eid]].tolist(),
            "button_body_velocities": velocities[self.rows[eid]].tolist(),
            "button_spring_anchors": [list(self._anchor_attribute(eid, color).Get()) for color in COLORS],
            "button_rest_world": self.rest_world[eid].tolist(),
            "button_outcome": deepcopy(self.last_outcome.get(eid, {})),
            "wrong_contact": bool(self.wrong_contact.get(eid, False)),
            "blocking_contacts": deepcopy(self.blocking_contacts.get(eid, []))} for eid in env_ids}


def create_backend(config, *, engine_factory=None, layout_sampler=None):
    from pressb.runtime.parallel import SharedWorldBackend
    return SharedWorldBackend(TableButtonsTask(config, layout_sampler=layout_sampler), engine_factory=engine_factory)
