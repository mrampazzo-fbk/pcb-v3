"""Time coordinates shared by design, model preparation and dataset export."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, kw_only=True)
class Simulation:
    horizon_seconds: float = 2200.
    native_step_seconds: float = 5.

    def __post_init__(self):
        self.stride(self.native_step_seconds)

    def stride(self, step):
        values = (self.horizon_seconds, self.native_step_seconds, step)
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError('horizon and time steps must be finite and positive')
        ratio = step / self.native_step_seconds
        frames = self.horizon_seconds / step
        if (not np.isclose(ratio, round(ratio), rtol=0, atol=1e-9)
                or ratio < 1 or not np.isclose(frames, round(frames), rtol=0, atol=1e-9)):
            raise ValueError('output step must be a multiple of the native step and divide the horizon')
        return round(ratio)

    @property
    def times(self):
        return np.arange(round(self.horizon_seconds / self.native_step_seconds) + 1) * self.native_step_seconds
