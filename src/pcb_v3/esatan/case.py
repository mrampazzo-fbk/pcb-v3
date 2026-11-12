"""Physical validation of the PCB experiment."""
import numpy as np
from ..time_grid import Simulation


def validate_solution(case, trial, node_weights, simulation=Simulation()):
    if not np.array_equal(case.times, simulation.times):
        raise ValueError(f'{trial.id}: incomplete or non-native TMD time grid')
    if not np.isfinite(case.temperature).all():
        raise ValueError(f'{trial.id}: invalid temperatures')
    missing = ({268, 99999} | set(node_weights)) - set(case.node_numbers)
    if missing:
        raise ValueError(f'{trial.id}: missing required nodes {sorted(missing)}')
    unit = case.get_temperature_unit()
    if unit not in ('Celsius', 'Kelvin'):
        raise ValueError(f'{trial.id}: unknown temperature units')
    temperatures = case.temperature - 273.15 if unit == 'Kelvin' else case.temperature
    for node, parameter in ((268, 'T_Interface'), (99999, 'T_enviroment')):
        observed = temperatures[case.node_numbers == node]
        if not np.allclose(observed, trial.scalars[parameter], rtol=0, atol=1e-3):
            raise ValueError(f'{trial.id}: boundary {parameter} differs')
    free = ~np.isin(case.node_numbers, (268, 99998, 99999))
    if not free.any() or not np.allclose(temperatures[free, 0], trial.scalars['T_enviroment'], rtol=0, atol=1e-3):
        raise ValueError(f'{trial.id}: initial free-node temperatures differ')
    # ESATAN may output the left limit at a discontinuity. Check every other
    # native sample, on every node, against its own original deck weight.
    check = ~np.any(np.isclose(case.times[:, None], np.array(trial.events)[:, 0], atol=1e-7, rtol=0), axis=1)
    expected = trial.scalars['Q_Heat_Source'] * trial.profile.power_at(case.times)
    weights = np.array([node_weights.get(int(n), 0.) for n in case.node_numbers])
    if not np.allclose(case.power[:, check], weights[:, None] * expected[check], rtol=1e-5, atol=1e-4):
        raise ValueError(f'{trial.id}: solver power differs from events/weights')
