"""Fixed binary schedules using the common scalar LHS and array Trial."""
from dataclasses import dataclass

import numpy as np

from ..time_grid import Simulation
from .design import HORIZON, PARAMETERS, scalar_lhs
from .profiles import Profile
from .variable import Trial

PROFILES = tuple(Profile(name, tuple(points)) for name, points in (
    ('off', ((0., 0.),)),
    ('continuous', ((0., 1.),)),
    ('cycle_standard', ((0., 1.), (240., 0.), (840., 1.), (1090., 0.),
                        (1100., 1.), (1340., 0.), (1940., 1.), (2190., 0.))),
    ('early_pulse', ((0., 1.), (240., 0.), (1100., 1.), (1340., 0.))),
    ('late_pulse', ((0., 0.), (840., 1.), (1090., 0.), (1940., 1.), (2190., 0.))),
    ('short_pulses', ((0., 0.), (100., 1.), (200., 0.), (900., 1.), (1000., 0.),
                       (1200., 1.), (1300., 0.), (2000., 1.), (2100., 0.))),
    ('long_duty', ((0., 1.), (400., 0.), (700., 1.), (1050., 0.),
                    (1100., 1.), (1500., 0.), (1800., 1.), (2150., 0.))),
    ('delayed_start', ((0., 0.), (1100., 1.), (1340., 0.), (1940., 1.), (2190., 0.))),
    ('early_shutdown', ((0., 1.), (240., 0.), (840., 1.), (1090., 0.), (1100., 1.), (1340., 0.))),
    ('alternating_orbits', ((0., 1.), (240., 0.), (840., 1.), (1090., 0.))),
    ('initial_off', ((0., 0.), (840., 1.), (1090., 0.), (1100., 1.), (1340., 0.), (1940., 1.), (2190., 0.))),
    ('initial_on', ((0., 1.), (300., 0.), (840., 1.), (1090., 0.),
                     (1100., 1.), (1340., 0.), (1940., 1.), (2190., 0.))),
))


@dataclass(frozen=True)
class Config(Simulation):
    samples: int = 500
    scalar_seed: int = 42
    profiles: tuple[Profile, ...] = PROFILES


def make_design(config=Config()):
    if (not config.profiles or len({p.id for p in config.profiles}) != len(config.profiles)
            or type(config.samples) is not int or config.samples < len(config.profiles)):
        raise ValueError('catalog requires distinct profiles and at least one sample per profile')
    for profile in config.profiles:
        if not profile.id or not all(c.isalnum() or c in '_-' for c in profile.id):
            raise ValueError('profile IDs must contain letters, numbers, underscores or hyphens')
        profile.validate(config.horizon_seconds)
    rng = np.random.default_rng(config.scalar_seed)
    scalars = scalar_lhs(config.samples, rng)[rng.permutation(config.samples)]
    size, remainder = divmod(config.samples, len(config.profiles))
    capacity = max(16, max(len(p.changes) for p in config.profiles) + 1)
    trials, offset = [], 0
    for group, profile in enumerate(config.profiles, 1):
        count = size + (group <= remainder)
        fraction = float(profile.interval_mean(np.array([0.]), config.horizon_seconds, config.horizon_seconds)[0])
        for i in range(offset, offset + count):
            trials.append(Trial(id=f'{profile.id}-s{i:03d}', group=group, pair=i,
                scalars=dict(zip(PARAMETERS, map(float, scalars[i]))), events=profile.changes,
                capacity=capacity, on_fraction=fraction, concentration=0.))
        offset += count
    return trials
