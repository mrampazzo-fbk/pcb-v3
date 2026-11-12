"""Immutable experiment definition, validated resume, one sweep and one HDF5."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

import numpy as np

from .generation import design
from .generation.study import document, generator, load_design
from .generation.design import PARAMETERS
from esatan_data.writer import export_fields
from .hdf5.power import export_blocks, field_metadata, add_total_power
from .esatan.case import validate_solution
from .esatan.model import BOUNDARY_NODES, DEFAULT_DECK, digest, prepare, prepared_deck, weights
from .esatan.launcher import solve
from esatan_data.collection import SolvedSweep
from .esatan.inputs import profile_points, to_case


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
        if any(p.name != '.lock' for p in work.iterdir()):
            raise ValueError('work directory is not empty; use a new directory')
        save_json(path, document(config, trials))
    fixed = family(config) == 'single'
    points = profile_points(trials[0] if fixed else max(trials, key=lambda t: t.group), config)
    period = config.period if fixed else (None if module is design else config.horizon_seconds + 1)
    deck = prepare(source, work / 'inputs', points,
                   period=period, simulation=config)
    return trials, deck


def checked_sweep(work):
    work = Path(work)
    config, trials = load_design(work / 'design.json')
    deck = prepared_deck(work / 'inputs')
    sweep = SolvedSweep(work / 'sweep', deck=deck)
    if [r.id for r in sweep.cases] != [t.id for t in trials]:
        raise ValueError('solved cases do not match design')
    node_weights = weights(deck)
    for i, ref, case in sweep.iter_cases():
        if ref.values != to_case(trials[i], config).values:
            raise ValueError(f'{ref.id}: tracker inputs differ')
        validate_solution(case, trials[i], node_weights, config)
    sweep.param_names = list(PARAMETERS)
    sweep.params = np.array([[t.scalars[p] for p in PARAMETERS] for t in trials])
    return sweep, trials, node_weights


def export(work, output=None, step=None):
    """Export a validated PCB dataset with interval-mean power and provenance."""
    work = Path(work).expanduser().resolve()
    with locked(work):
        config, _ = load_design(work / 'design.json')
        return _export(work, output, step, config, checked_sweep(work))


def _export(work, output, step, config, solved):
    step = config.native_step_seconds if step is None else step
    stride = config.stride(step)
    sweep, trials, node_weights = solved
    output = output or work / dataset_name(config, step)

    def finalize(h5):
        add_total_power(h5)
        h5.attrs.update(format='pcb-v3', dataset_kind='validated_pcb',
                        design_sha256=digest(work / 'design.json'),
                        model_bundle_sha256=digest(work / 'inputs/bundle.json'),
                        native_step_seconds=config.native_step_seconds,
                        horizon_seconds=config.horizon_seconds, output_step_seconds=step,
                        load_sampling="forward_interval_mean",
                        frozen_properties="Capacitance(0), GL(0)")

    return export_fields(sweep, output, stride=stride, boundary_nodes=BOUNDARY_NODES,
                          finalize=finalize, blocks=export_blocks(trials, node_weights, step, config),
                          field_attrs=field_metadata(step),
                          parameter_units={"T_enviroment":"degC", "T_Interface":"degC", "Q_Heat_Source":"W"})


def dataset_name(config, step):
    kind = family(config)
    stem = 'profile-catalog' if kind == 'catalog' else f'{kind}-profile'
    return f'pcb-{stem}-{step:g}s.h5'


def run(work, config, install, source=DEFAULT_DECK):
    work = Path(work).resolve()
    with locked(work):
        trials, deck = initialize(work,config,source)
        directory = work / 'sweep'
        solve(install, deck, trials, directory, weights(deck), config)
        solved = checked_sweep(work)
        output = work / dataset_name(config, config.native_step_seconds)
        if not output.exists():
            _export(work, output, None, config, solved)
        return output
