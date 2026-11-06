import abc
from typing import Dict

import numpy as np


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict, prev_action: np.ndarray | None = None, is_rtc: bool = False) -> Dict:
        """Infer actions from observations.

        Args:
            obs: Observation dictionary.
            prev_action: Previous action chunk for guided inference (RTC mode).
            is_rtc: Whether to use Real-Time Action Chunking mode.
        """

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
