import logging
import threading
import time
from typing import Dict

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy

logger = logging.getLogger(__name__)


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.

    When is_rtc=True, uses Real-Time Action Chunking (RTC) with a background
    inference thread that overlaps computation with execution.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        is_rtc: bool = False,
        s: int = 16,
        d: int = 8,
    ):
        self._policy = policy
        self._action_horizon = action_horizon
        self._is_rtc = is_rtc
        self._s = s
        self._d = d

        self._cur_step: int = 0
        self._last_results: Dict[str, np.ndarray] | None = None
        self._last_actions: np.ndarray | None = None  # Raw actions for RTC guidance

        # RTC background inference state
        self._obs: Dict | None = None
        self._background_results: Dict[str, np.ndarray] | None = None
        self._background_running: bool = False

        if self._is_rtc:
            self._infer_thread = threading.Thread(target=self._background_infer, daemon=True)
            self._infer_thread.start()

    def _background_infer(self):
        """Background thread for RTC: triggers inference at step s."""
        while True:
            if self._cur_step == self._s and self._obs is not None:
                self._background_running = True
                try:
                    prev_action = self._last_actions if self._last_actions is not None else None
                    self._background_results = self._policy.infer(
                        self._obs, prev_action=prev_action, is_rtc=True
                    )
                except Exception as e:
                    logger.error(f"Background inference error: {e}")
                    self._background_results = None
                self._background_running = False
            else:
                time.sleep(0.001)

    @override
    def infer(self, obs: Dict, prev_action=None, is_rtc: bool = False) -> Dict:  # noqa: UP006
        if self._is_rtc:
            return self._infer_rtc(obs)
        else:
            return self._infer_standard(obs)

    def _infer_standard(self, obs: Dict) -> Dict:
        """Standard (non-RTC) action chunking."""
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            # Squeeze batch dimension: (1, H, D) -> (H, D)
            action = self._last_results["action"]
            if isinstance(action, np.ndarray) and action.ndim == 3:
                self._last_results["action"] = action[0]
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    def _infer_rtc(self, obs: Dict) -> Dict:
        """RTC action chunking with background inference."""
        # Initial inference
        if self._last_results is None:
            self._last_results = self._policy.infer(obs, prev_action=None, is_rtc=True)
            action = self._last_results["action"]
            if action.ndim == 3:
                action = action[0]
            self._last_actions = action.copy()
            self._last_results = {"action": action}
            self._cur_step = 0

        # Return current step's action
        results = tree.map_structure(lambda x: x[self._cur_step, ...], self._last_results)
        self._obs = obs
        self._cur_step += 1

        # At step s+d, wait for background inference and swap results
        if self._cur_step == self._s + self._d:
            while self._background_running:
                time.sleep(0.001)
            if self._background_results is not None:
                action = self._background_results["action"]
                if action.ndim == 3:
                    action = action[0]
                self._last_actions = action.copy()
                self._last_results = {"action": action}
                self._cur_step -= self._s
                self._background_results = None
            else:
                # Background inference failed, reset for fresh inference
                self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._last_actions = None
        self._cur_step = 0
        self._obs = None
        self._background_results = None
