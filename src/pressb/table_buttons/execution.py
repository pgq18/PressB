"""Contact-limited execution of scripted table-button motion plans."""
from __future__ import annotations

import numpy as np


_DEFAULT_PHASE_TIMES = {
    'press_start': 3.1,
    'release_start': 4.45,
    'return_hover_end': 5.55,
}


def press_phase_times(plan_report):
    """Read generated phase times, with historical defaults for old plans."""
    values = plan_report.get('phase_times', _DEFAULT_PHASE_TIMES)
    times = {name: float(values[name]) for name in _DEFAULT_PHASE_TIMES}
    if not all(np.isfinite(value) for value in times.values()):
        raise ValueError('Press phase times must be finite')
    if not 0 <= times['press_start'] < times['release_start'] < times['return_hover_end']:
        raise ValueError('Press, release and return phases must be strictly ordered')
    return times


def in_press_window(seconds, phase_times):
    return phase_times['press_start'] <= seconds < phase_times['release_start']


def contact_limited_command(seconds, nominal, held_command, path_times, path_commands, phase_times):
    """Hold measured-depth contact, then retreat to the generated hover target.

    This changes only the robot target, never button state. The caller chooses
    its existing execution frequency (physics rate for demos, 30 Hz for data).
    """
    if held_command is None:
        return nominal
    if in_press_window(seconds, phase_times):
        return held_command.copy()
    release = phase_times['release_start']
    hover_end = phase_times['return_hover_end']
    if release <= seconds < hover_end:
        u = (seconds-release)/(hover_end-release)
        blend = u*u*u*(10+u*(-15+6*u))
        hover_command = np.array([
            np.interp(hover_end, path_times, path_commands[:, j])
            for j in range(path_commands.shape[1])])
        return held_command+(hover_command-held_command)*blend
    return nominal
