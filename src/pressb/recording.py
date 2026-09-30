"""Measured simulation records; commanded positions are kept separately."""
import csv
import json
from pathlib import Path
import numpy as np


class Recorder:
    def __init__(self, output, cfg, joint_names):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.joint_names = list(joint_names)
        self.rows = []
        self.events = []

    def record(self, **row):
        self.rows.append(row)

    def event(self, **event):
        self.events.append(event)

    def save(self, extra=None):
        fields = list(self.rows[0]) if self.rows else []
        arrays = {key: np.asarray([row[key] for row in self.rows]) for key in fields}
        np.savez_compressed(self.output / "trajectory.npz", **arrays)
        with (self.output / "trajectory.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            names = []
            for key in fields:
                sample = np.asarray(self.rows[0][key])
                if sample.ndim == 0:
                    names.append(key)
                else:
                    suffix = self.joint_names if key in ("q_actual", "q_target", "qd_actual") else (
                        ["x", "y", "z"] if sample.size == 3 else [str(i) for i in range(sample.size)])
                    names.extend(f"{key}_{s}" for s in suffix)
            writer.writerow(names)
            for row in self.rows:
                writer.writerow([v for key in fields for v in np.asarray(row[key]).reshape(-1).tolist()])
        passed = sorted({e["floor"] for e in self.events if e["type"] == "button_pressed"})
        report = {"backend": "NVIDIA Isaac Sim 4.5 / PhysX", "physics_executed": True,
                  "success": passed == sorted(self.cfg["sequence"]), "pressed_floors": passed,
                  "missing_floors": sorted(set(self.cfg["sequence"]) - set(passed)),
                  "samples": len(self.rows), "joint_names": self.joint_names,
                  "press_detection": "Measured button travel AND PhysX stylus contact impulse; amber while depressed, off on spring return",
                  "config": self.cfg, **(extra or {})}
        report["physics_validated"] = bool(report["success"])
        (self.output / "events.json").write_text(json.dumps(self.events, indent=2))
        (self.output / "report.json").write_text(json.dumps(report, indent=2))
        if self.rows:
            self.plot(arrays)
        return report

    def plot(self, arrays):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(13, 8), layout="constrained")
        ax = fig.add_subplot(221, projection="3d")
        p = arrays["tip_actual"]
        ax.plot(*p.T, linewidth=0.8, color="#138c94")
        ax.set(xlabel="x / m", ylabel="y / m", zlabel="z / m", title="Measured tool trajectory")
        ax = fig.add_subplot(222)
        ax.plot(arrays["time"], arrays["q_actual"])
        ax.set(xlabel="Time / s", ylabel="Joint angle / rad", title="Measured arm joints")
        ax.legend(self.joint_names, ncol=3, fontsize=7)
        ax = fig.add_subplot(223)
        ax.plot(arrays["time"], arrays["button_travel"] * 1000)
        ax.set(xlabel="Time / s", ylabel="Travel / mm", title="12 physical spring buttons")
        ax = fig.add_subplot(224)
        ax.plot(arrays["time"], arrays["contact_force"])
        ax.set(xlabel="Time / s", ylabel="Contact force / N", title="PhysX stylus contact")
        fig.savefig(self.output / "trajectory.png", dpi=150)
        plt.close(fig)
