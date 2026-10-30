"""Resolve PAR case order once and stream aligned TMD results."""
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from esatan_launcher import Tracker
from esatan_launcher.deck import parse_deck
from esatan_results_reader import EsatanTMD

from .case import TmdCase


def read_case(path):
    return TmdCase(EsatanTMD(fname=path))


def attempt_directory(directory, path):
    """Resolve local attempts after moving a study, preserving live external refs."""
    if path is None:
        return None
    path = Path(path)
    candidate = Path(directory) / path.name
    return candidate if not path.exists() and candidate.exists() else path


def load_tracker(directory):
    directory = Path(directory)
    tracker = Tracker.load(directory / 'tracker.json')
    deck = directory.parent / 'inputs' / Path(tracker.deck).name
    if not Path(tracker.deck).exists() and deck.exists():
        tracker.deck = deck
    for record in tracker.records:
        record.attempt_dir = attempt_directory(directory, record.attempt_dir)
    if tracker.build_dir and not Path(tracker.build_dir).exists():
        candidate = directory / 'build' / Path(tracker.build_dir).name
        if candidate.exists():
            tracker.build_dir = candidate
    return tracker


def par_ids(path):
    return re.findall(r"!INITIAL\s*=\s*'([^']+)'", Path(path).read_text())


def attempt_paths(tracker, attempt):
    model = parse_deck(tracker.deck).model_name
    return {case_id: (i, attempt / f'{model}.TMD{i if i > 1 else ""}')
            for i, case_id in enumerate(par_ids(attempt / f'{tracker.model}.PAR'), 1)}


@dataclass(frozen=True)
class CaseRef:
    id: str
    values: dict
    attempt_dir: Path
    tmd_path: Path
    index_in_attempt: int


class SolvedSweep:
    def __init__(self, directory):
        tracker = load_tracker(directory)
        if not tracker.records or any(r.status != 'done' for r in tracker.records):
            raise ValueError('sweep is empty or incomplete')
        self.model = tracker.model
        attempts = {a: attempt_paths(tracker, a) for a in {r.attempt_dir for r in tracker.records}}
        self.cases = tuple(CaseRef(r.case.id, dict(r.case.values), r.attempt_dir,
                                  attempts[r.attempt_dir][r.case.id][1], attempts[r.attempt_dir][r.case.id][0])
                           for r in tracker.records)
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
