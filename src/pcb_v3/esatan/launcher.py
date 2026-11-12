"""Execute PCB cases through the public launcher API."""
from esatan_results_reader.esatan_tmd import ThermalResultError

from ..time_grid import Simulation
from .case import validate_solution
from .inputs import to_case
from esatan_data.collection import read_case


def solve(install, deck, trials, directory, node_weights, simulation=Simulation()):
    expected = {trial.id: trial for trial in trials}

    def valid_result(case, path):
        try:
            validate_solution(read_case(path), expected[case.id], node_weights, simulation)
        except (OSError, ValueError, ThermalResultError):
            return False
        return True

    result = install.sweep(deck, [to_case(trial, simulation) for trial in trials],
                           sweep_dir=directory, outputs=('T250', 'QI250'),
                           include_swept_entities=False, validate_result=valid_result)
    if not result.ok:
        raise RuntimeError(f'incomplete sweep: {directory}; inspect tracker and solver logs')
    return result
