"""Shared scalar sampling, paired timing histories and fixed periodic profiles."""
from dataclasses import dataclass

import numpy as np

from ..time_grid import Simulation
from .profiles import Profile, repeat_events

HORIZON = Simulation().horizon_seconds
CAPACITY = 16
PARAMETERS = ('T_enviroment', 'T_Interface', 'Q_Heat_Source')
BOUNDS = np.array([(10., 40.), (10., 50.), (10., 30.)])


def scalar_lhs(n, rng, bounds=BOUNDS):
    """One shared scalar sampler; callers can continue using the same RNG."""
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (len(PARAMETERS), 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError('expected finite increasing bounds for all scalar parameters')
    unit = np.column_stack([(rng.permutation(n) + rng.random(n)) / n for _ in PARAMETERS])
    return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])


@dataclass(frozen=True)
class Config(Simulation):
    scalar_seed: int = 42
    timing_seed: int = 42
    min_on: float = 30.0
    min_off: float = 30.0
    samples: int = 50
    source_profile: tuple[tuple[float, float], ...] = ()
    period: float = 0.0
    scalar_values: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class Trial:
    id: str
    group: int
    pair: int
    scalars: dict[str, float]
    events: tuple[tuple[float, float], ...]
    array_points: tuple[tuple[float, float], ...] = ()

    @property
    def profile(self):
        return Profile(self.id, self.events)


def make_design(config=Config()):
    if not isinstance(config.samples, int) or config.samples < 1:
        raise ValueError('samples must be a positive integer')
    minima = np.array([config.min_off, config.min_on] * 6 + [config.min_off])
    if not config.source_profile and (not np.isfinite(minima).all() or np.any(minima <= 0) or minima.sum() >= config.horizon_seconds):
        raise ValueError('positive minimum durations must leave slack for six intervals')
    rng = np.random.default_rng(config.scalar_seed)
    n = config.samples
    scalars = scalar_lhs(n, rng)
    if config.scalar_values is not None:
        scalars = np.asarray([config.scalar_values], dtype=float)
        if n != 1 or scalars.shape != (1, len(PARAMETERS)) or not np.isfinite(scalars).all() or scalars[0, 2] < 0:
            raise ValueError('scalar_values requires one case with finite temperatures and nonnegative power')
    if config.source_profile:
        points = np.asarray(config.source_profile, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all()
                or points[0, 0] != 0 or np.any(np.diff(points[:, 0]) <= 0)
                or config.period <= points[-1, 0] or not np.isfinite(config.period)
                or not np.any(points[:, 1] > 0)):
            raise ValueError('invalid source profile or period')
        on = points[:, 1].max()
        if not np.isin(points[:, 1], (0., on)).all():
            raise ValueError('source profile must be binary on/off')
        normalized = tuple((float(t), float(level / on)) for t, level in points)
        events = repeat_events(normalized, config.period, config.horizon_seconds)
        Profile('source', events).validate(config.horizon_seconds, allow_endpoint=True)
        group = sum(level == 1 and t < config.horizon_seconds for t, level in events)
        return [Trial(f'fixed-p{pair:03d}', group, pair, dict(zip(PARAMETERS, values)),
                      events, normalized) for pair, values in enumerate(scalars)]
    rng = np.random.default_rng(config.timing_seed)
    trials = []
    for group in range(1, 7):
        minimum = minima[:2 * group + 1]
        for pair, values in enumerate(scalars):
            durations = minimum + rng.dirichlet(np.ones(len(minimum))) * (config.horizon_seconds - minimum.sum())
            times = np.r_[0., np.cumsum(durations)[:-1]]
            events = tuple((float(t), float(i % 2)) for i, t in enumerate(times))
            trials.append(Trial(f'g{group}-p{pair:02d}', group, pair, dict(zip(PARAMETERS, values)), events))
    for group in range(1,7):
        if len({t.events for t in trials if t.group == group}) != n:
            raise ValueError('timing histories must be distinct within each group')
    return trials
