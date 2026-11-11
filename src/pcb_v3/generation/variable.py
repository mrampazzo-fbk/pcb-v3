"""Stratified on-time and independent on/off allocations for variable profiles."""
from dataclasses import dataclass

import numpy as np

from ..time_grid import Simulation
from .design import HORIZON, PARAMETERS, Trial as BaseTrial, scalar_lhs


@dataclass(frozen=True)
class Config(Simulation):
    scalar_seed: int = 42
    timing_seed: int = 42
    samples: int = 50  # Per interval count; the same scalar LHS is paired across groups.
    intervals: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    min_on: float = 30.
    min_off: float = 30.
    on_fraction: tuple[float, float] = (0., 1.)
    concentrations: tuple[float, ...] = (0.3, 1., 10.)
    total_samples: int | None = None  # One independent scalar LHS across all groups.
    always_on: int = 0
    always_off: int = 0
    scalar_bounds: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True)
class Trial(BaseTrial):
    capacity: int = 16
    on_fraction: float = 0.
    concentration: float = 1.

    @property
    def profile_kind(self):
        return {-1: 'always_on', 0: 'always_off'}.get(self.group, 'variable')


def group_counts(config):
    """Split the variable cases evenly; put any remainder in the first groups."""
    if any(type(n) is not int or n < 0 for n in (config.always_on, config.always_off)):
        raise ValueError('always_on and always_off must be nonnegative integers')
    if config.total_samples is None:
        if config.always_on or config.always_off:
            raise ValueError('constant profiles require total_samples')
        return [config.samples] * len(config.intervals)
    if type(config.total_samples) is not int or config.total_samples < 1:
        raise ValueError('total_samples must be a positive integer')
    if config.samples != 50:
        raise ValueError('samples and total_samples cannot be combined')
    variable = config.total_samples - config.always_on - config.always_off
    if variable < len(config.intervals):
        raise ValueError('total_samples must leave at least one case per interval count')
    n, remainder = divmod(variable, len(config.intervals))
    return [n + (i < remainder) for i in range(len(config.intervals))]


def feasible_fractions(config, k):
    """Intersection of the requested range with the physical duration constraints."""
    low = max(config.on_fraction[0], k * config.min_on / config.horizon_seconds)
    high = min(config.on_fraction[1], 1 - (k + 1) * config.min_off / config.horizon_seconds)
    if low >= high:
        raise ValueError(f'{k} intervals: no nonempty on-fraction range compatible '
                         f'with the minimum on/off durations and {config.horizon_seconds:g} s horizon')
    return low, high


def make_design(config=Config()):
    if type(config.samples) is not int or config.samples < 1:
        raise ValueError('samples must be a positive integer per group')
    if (not config.intervals or any(type(k) is not int or k < 1 for k in config.intervals)
            or len(set(config.intervals)) != len(config.intervals)):
        raise ValueError('intervals must be distinct positive integers')
    if not np.isfinite([config.min_on, config.min_off]).all() or min(config.min_on, config.min_off) <= 0:
        raise ValueError('minimum on/off durations must be finite and positive')
    if (len(config.on_fraction) != 2 or not np.isfinite(config.on_fraction).all()
            or not 0 <= config.on_fraction[0] < config.on_fraction[1] <= 1):
        raise ValueError('on_fraction requires 0 <= lower < upper <= 1')
    if (not config.concentrations or not np.isfinite(config.concentrations).all()
            or min(config.concentrations) <= 0):
        raise ValueError('concentrations must be finite and positive')
    ranges = {k: feasible_fractions(config, k) for k in config.intervals}
    counts = group_counts(config)
    n = config.samples if config.total_samples is None else config.total_samples
    capacity = max(16, 2 * max(config.intervals) + 2)
    rng = np.random.default_rng(config.scalar_seed)
    scalars = scalar_lhs(n, rng) if config.scalar_bounds is None else scalar_lhs(n, rng, config.scalar_bounds)
    if config.total_samples is not None:
        scalars = scalars[rng.permutation(n)]
    rng = np.random.default_rng(config.timing_seed)
    trials = []
    offset = 0
    for k, n in zip(config.intervals, counts):
        group_scalars = scalars if config.total_samples is None else scalars[offset:offset + n]
        low, high = ranges[k]
        # Stratification explicitly covers low, medium and high total on-time.
        fractions = low + (rng.permutation(n) + rng.random(n)) / n * (high - low)
        # Balanced counts, independently shuffled relative to fraction strata.
        alpha_indices = rng.permutation(n) % len(config.concentrations)
        for local_index, (values, fraction, alpha_index) in enumerate(zip(group_scalars, fractions, alpha_indices)):
            pair = local_index if config.total_samples is None else offset + local_index
            alpha = config.concentrations[int(alpha_index)]
            total_on = config.horizon_seconds * fraction
            on = config.min_on + rng.dirichlet(np.full(k, alpha)) * (total_on - k * config.min_on)
            off = config.min_off + rng.dirichlet(np.full(k + 1, alpha)) * (
                config.horizon_seconds - total_on - (k + 1) * config.min_off)
            durations = np.empty(2 * k + 1)
            durations[::2], durations[1::2] = off, on
            times = np.r_[0., np.cumsum(durations)[:-1]]
            events = tuple((float(t), float(i % 2)) for i, t in enumerate(times))
            trial = Trial(id=f'v2-g{k}-p{pair:03d}', group=k, pair=pair,
                          scalars=dict(zip(PARAMETERS, map(float, values))), events=events,
                          capacity=capacity, on_fraction=float(fraction), concentration=float(alpha))
            trial.profile.validate(config.horizon_seconds)
            trials.append(trial)
        offset += n
    for kind, group, state, count in (('always_on', -1, 1., config.always_on),
                                       ('always_off', 0, 0., config.always_off)):
        for pair in range(offset, offset + count):
            trials.append(Trial(id=f'v2-{kind}-s{pair:03d}', group=group, pair=pair,
                                scalars=dict(zip(PARAMETERS, map(float, scalars[pair]))),
                                events=((0., state),), capacity=capacity,
                                on_fraction=state, concentration=0.))
        offset += count
    return trials
