"""Immutable experiment definition, validated resume, one sweep and one HDF5."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

import numpy as np
from esatan_results_reader.esatan_tmd import ThermalResultError

from .generation import design
from .generation.design import PARAMETERS, HORIZON
from .hdf5.writer import export_dataset
from .hdf5.power import add_forcing
from .esatan.case import validate_solution
from .esatan.model import BOUNDARY_NODES, DEFAULT_DECK, prepare, weights
from .esatan.launcher import solve
from .esatan.sweep import SolvedSweep, attempt_paths, read_case, load_tracker


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


@contextmanager
def locked(work):
    work.mkdir(parents=True, exist_ok=True)
    with (work / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def generator(config):
    from .generation import catalog, variable
    for module in (design, variable, catalog):
        if isinstance(config, module.Config):
            return module
    raise ValueError('unsupported study configuration')


def load_design(path):
    from .generation import catalog, variable
    schema = json.loads(Path(path).read_text()).get('schema')
    modules = {'pcb-variable-power-v1': design, 'pcb-variable-power-v2': variable,
               'pcb-catalog-v1': catalog}
    if schema not in modules:
        raise ValueError('unsupported study schema')
    return modules[schema].load_design(path)


def family(config):
    name = generator(config).__name__.rsplit('.', 1)[-1]
    if name == 'catalog':
        return 'catalog'
    return 'single' if name == 'design' and config.source_profile else 'variable'


def initialize(work, config, source=DEFAULT_DECK):
    module = generator(config)
    trials = module.make_design(config)
    path = work / 'design.json'
    if path.exists():
        if load_design(path) != (config, trials):
            raise ValueError('study configuration differs; use a new directory')
    else:
        save_json(path, module.document(config, trials))
    fixed = family(config) == 'single'
    points = trials[0].points if fixed else max(trials, key=lambda t: t.group).points
    period = config.period if fixed else (None if module is design else HORIZON + 1)
    deck = prepare(source, work / 'inputs', points,
                   period=period)
    return trials, deck


def refresh_tracker(directory, trials, node_weights):
    """Recover complete TMDs after interruption; never trust CSV status alone.

    PAR order supplies the suffix even when an attempt has failed/pending cases.
    Incomplete outputs stay on disk and are retried in a new launcher attempt.
    """
    path = directory / 'tracker.json'
    if not path.exists():
        return
    tracker = load_tracker(directory)
    expected = {t.id: t for t in trials}
    records = {r.case.id: r for r in tracker.records}
    if list(records) != list(expected) or any(dict(r.case.values) != expected[r.case.id].case().values for r in tracker.records):
        raise ValueError('tracker does not match study selection')
    attempts = set(directory.glob('attempt-*')) | {r.attempt_dir for r in tracker.records if r.attempt_dir}
    for attempt in sorted(attempts):
        pars = list(attempt.glob('*.PAR'))
        if len(pars) != 1:
            continue
        for case_id, (index, tmd) in attempt_paths(tracker, attempt).items():
            record = records[case_id]
            if record.status == 'done' and record.attempt_dir != attempt:
                continue
            try:
                case = read_case(tmd)
                validate_solution(case, expected[case_id], node_weights)
            except (OSError, ValueError, ThermalResultError, IndexError, KeyError):
                if record.status == 'done':
                    record.status = 'pending'
                continue
            record.status, record.attempt_dir = 'done', attempt
    tracker.save()

def checked_sweep(work):
    _, trials = load_design(work / 'design.json')
    sweep = SolvedSweep(work / 'sweep')
    if [r.id for r in sweep.cases] != [t.id for t in trials]:
        raise ValueError('solved cases do not match design')
    node_weights = weights(next((work / 'inputs').glob('*.d')))
    for i, ref, case in sweep.iter_cases():
        if ref.values != trials[i].case().values:
            raise ValueError(f'{ref.id}: tracker inputs differ')
        validate_solution(case,trials[i],node_weights)
    sweep.param_names = list(PARAMETERS)
    sweep.params = np.array([[t.scalars[p] for p in PARAMETERS] for t in trials])
    return sweep, trials, node_weights


def export(work, output, step=5, *, solved=None):
    if step not in (5,50):
        raise ValueError('output step must be 5 or 50 seconds')
    sweep, trials, node_weights = solved or checked_sweep(work)
    return export_dataset(sweep,output,stride=step//5,boundary_nodes=BOUNDARY_NODES,
                          finalize=lambda h5: add_forcing(h5,trials,node_weights,step))


def dataset_name(config, step):
    kind = family(config)
    stem = 'profile-catalog' if kind == 'catalog' else f'{kind}-profile'
    return f'pcb-{stem}-{step}s.h5'


def run(work, config, install, source=DEFAULT_DECK):
    work = Path(work).resolve()
    with locked(work):
        trials, deck = initialize(work,config,source)
        directory = work / 'sweep'
        refresh_tracker(directory,trials,weights(deck))
        solve(install,deck,[t.case() for t in trials],directory)
        solved = checked_sweep(work)
        output = work / dataset_name(config, 5)
        if not output.exists():
            export(work,output,solved=solved)
        return output
