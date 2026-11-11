"""Translate a scientific trial into the PCB model's ESATAN CHANGE targets."""
import numpy as np
from esatan_launcher import Case

from ..time_grid import Simulation
from .acd import ARRAY_NAME


def profile_points(trial, simulation=Simulation()):
    if trial.array_points:
        return np.asarray(trial.array_points)
    capacity = getattr(trial, 'capacity', 16)
    extra = capacity - len(trial.events)
    if extra < 1:
        raise ValueError('profile array needs room for its final point')
    tail = np.linspace(trial.events[-1][0], simulation.horizon_seconds, extra + 1)[1:]
    points = np.array((*trial.events, *((float(t), trial.events[-1][1]) for t in tail)))
    if np.any(np.diff(points[:, 0]) <= 0):
        raise ValueError('profile array times must be strictly increasing')
    return points


def to_case(trial, simulation=Simulation()):
    values = dict(trial.scalars)
    if not trial.array_points:
        values.update({f'{ARRAY_NAME}({axis},{i})': float(value)
                       for i, point in enumerate(profile_points(trial, simulation), 1)
                       for axis, value in enumerate(point, 1)})
    return Case(trial.id, values)
