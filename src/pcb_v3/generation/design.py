"""Shared LHS/Trial, fixed deck profiles and loading of saved v1 studies."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from esatan_launcher import Case

from ..esatan.acd import ARRAY_NAME
from .profiles import Profile, repeat_events

HORIZON = 2200.0
CAPACITY = 16
PARAMETERS = ('T_enviroment', 'T_Interface', 'Q_Heat_Source')
BOUNDS = np.array([(10., 40.), (10., 50.), (10., 30.)])


def scalar_lhs(n, rng):
    """One shared scalar sampler; callers can continue using the same RNG."""
    unit = np.column_stack([(rng.permutation(n) + rng.random(n)) / n for _ in PARAMETERS])
    return BOUNDS[:, 0] + unit * (BOUNDS[:, 1] - BOUNDS[:, 0])


@dataclass(frozen=True)
class Config:
    scalar_seed: int = 42
    timing_seed: int = 42
    min_on: float = 30.0
    min_off: float = 30.0
    samples: int = 50
    source_profile: tuple[tuple[float, float], ...] = ()
    period: float = 0.0


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

    @property
    def points(self):
        if self.array_points:
            return np.asarray(self.array_points)
        # Fill the final off segment: redundant points never introduce a jump.
        extra = CAPACITY - len(self.events)
        tail = np.linspace(self.events[-1][0], HORIZON, extra + 1)[1:]
        return np.array((*self.events, *((float(t), 0.) for t in tail)))

    def case(self):
        values = dict(self.scalars)
        if not self.array_points:
            values.update({f'{ARRAY_NAME}({axis},{i})': float(value)
                           for i, point in enumerate(self.points, 1)
                           for axis, value in enumerate(point, 1)})
        return Case(self.id, values)


def make_design(config=Config()):
    if not isinstance(config.samples, int) or config.samples < 1:
        raise ValueError('samples must be a positive integer')
    minima = np.array([config.min_off, config.min_on] * 6 + [config.min_off])
    if not config.source_profile and (not np.isfinite(minima).all() or np.any(minima <= 0) or minima.sum() >= HORIZON):
        raise ValueError('positive minimum durations must leave slack for six intervals')
    rng = np.random.default_rng(config.scalar_seed)
    n = config.samples
    scalars = scalar_lhs(n, rng)
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
        events = repeat_events(normalized, config.period, HORIZON)
        Profile('source', events).validate(HORIZON, allow_endpoint=True)
        group = sum(level == 1 and t < HORIZON for t, level in events)
        return [Trial(f'fixed-p{pair:03d}', group, pair, dict(zip(PARAMETERS, values)),
                      events, normalized) for pair, values in enumerate(scalars)]
    rng = np.random.default_rng(config.timing_seed)
    trials = []
    for group in range(1, 7):
        minimum = minima[:2 * group + 1]
        for pair, values in enumerate(scalars):
            durations = minimum + rng.dirichlet(np.ones(len(minimum))) * (HORIZON - minimum.sum())
            times = np.r_[0., np.cumsum(durations)[:-1]]
            events = tuple((float(t), float(i % 2)) for i, t in enumerate(times))
            trials.append(Trial(f'g{group}-p{pair:02d}', group, pair, dict(zip(PARAMETERS, values)), events))
    for group in range(1,7):
        if len({t.events for t in trials if t.group == group}) != n:
            raise ValueError('timing histories must be distinct within each group')
    return trials


def document(config, trials):
    settings = asdict(config)
    if not config.source_profile:
        for name in ('source_profile', 'period'):
            settings.pop(name)
    if config.samples == 50:
        settings.pop('samples')
    cases = []
    for trial in trials:
        row = asdict(trial)
        if not trial.array_points:
            row.pop('array_points')
        cases.append(row)
    return {'schema': 'pcb-variable-power-v1', 'config': settings,
            'horizon_seconds': HORIZON, 'array_capacity': len(trials[0].points) if trials else CAPACITY,
            'cases': cases}


def load_design(path):
    doc = json.loads(Path(path).read_text())
    if doc['schema'] != 'pcb-variable-power-v1':
        raise ValueError('unsupported study schema')
    settings = doc['config'].copy()
    if 'source_profile' in settings:
        settings['source_profile'] = tuple(map(tuple, settings['source_profile']))
    config = Config(**settings)
    trials = [Trial(**{**row, 'events': tuple(map(tuple, row['events'])),
                       'array_points': tuple(map(tuple, row.get('array_points', ())))})
              for row in doc['cases']]
    if doc != json.loads(json.dumps(document(config, make_design(config)))):
        raise ValueError('design does not match its seeds/configuration')
    return config, trials
