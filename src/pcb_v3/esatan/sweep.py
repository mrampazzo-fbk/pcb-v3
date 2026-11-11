"""Stream and align the case artifacts identified by esatan-launcher."""
from collections import OrderedDict
from pathlib import Path

import numpy as np
from esatan_launcher import Tracker
from esatan_results_reader import EsatanTMD

from .case import TmdCase


def read_case(path):
    return TmdCase(EsatanTMD(fname=path))


class SolvedSweep:
    def __init__(self, directory, *, deck=None):
        tracker = Tracker.load(Path(directory) / 'tracker.json', deck=deck)
        self.model = tracker.model
        self.cases = tracker.completed_results()
        self.param_names = sorted(self.cases[0].values)
        self.params = np.array([[r.values[n] for n in self.param_names] for r in self.cases])
        self._cache = OrderedDict()

    @property
    def n_cases(self):
        return len(self.cases)

    @property
    def n_nodes(self):
        return len(self.first.nodes)

    @property
    def first(self):
        return self.case(0)

    def times(self, stride=1):
        return self.first.times[::stride]

    def case(self, index):
        path = self.cases[index].tmd_path
        if path not in self._cache:
            self._cache[path] = read_case(path)
        self._cache.move_to_end(path)
        while len(self._cache) > 2:
            self._cache.popitem(last=False)
        return self._cache[path]

    def validate_case(self, case, case_id):
        case.validate()
        first = self.first
        for name in ('node_numbers', 'node_labels', 'node_feature_names', 'times'):
            if not np.array_equal(getattr(case, name), getattr(first, name)):
                raise ValueError(f'{case_id}: {name} differs from the first case')
        if case.get_temperature_unit() != first.get_temperature_unit():
            raise ValueError(f'{case_id}: temperature units differ')
        if case.conductor_types != first.conductor_types:
            raise ValueError(f'{case_id}: conductor types differ')
        for kind in first.conductor_types:
            if not np.array_equal(case.conductors[f'{kind}_edges'], first.conductors[f'{kind}_edges']):
                raise ValueError(f'{case_id}: {kind} topology differs')

    def iter_cases(self):
        for i, ref in enumerate(self.cases):
            case = self.case(i)
            self.validate_case(case, ref.id)
            yield i, ref, case
