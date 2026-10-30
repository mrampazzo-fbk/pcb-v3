"""Finite binary histories and exact forward-interval integrals."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Profile:
    id: str
    changes: tuple[tuple[float, float], ...]

    def validate(self, horizon, *, allow_endpoint=False):
        points = np.asarray(self.changes, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
                or not np.isfinite(points).all() or points[0, 0] != 0
                or np.any(np.diff(points[:, 0]) <= 0)
                or (points[-1, 0] > horizon if allow_endpoint else points[-1, 0] >= horizon)
                or not np.isin(points[:, 1], [0, 1]).all()):
            raise ValueError(f'{self.id}: expected increasing binary events from zero before horizon')

    def power_at(self, times):
        starts, levels = np.array(self.changes).T
        return levels[np.searchsorted(starts, times, side='right') - 1]

    def interval_mean(self, times, duration, horizon):
        """Exact average on [t,t+duration), with zero beyond the horizon."""
        times = np.asarray(times)
        starts, levels = np.array(self.changes).T
        ends = np.r_[starts[1:], horizon]
        overlap = np.maximum(0., np.minimum(times[:, None] + duration, ends)
                             - np.maximum(times[:, None], starts))
        return overlap @ levels / duration


def repeat_events(points, period, horizon):
    """Expand one source cycle, including a restart exactly at the horizon."""
    return tuple((float(t + cycle * period), float(state))
                 for cycle in range(int(horizon // period) + 1)
                 for t, state in points if t + cycle * period <= horizon)
