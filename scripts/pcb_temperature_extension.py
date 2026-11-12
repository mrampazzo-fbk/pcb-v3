#!/usr/bin/env python3
"""Generate the 500-case variable-profile study with 100–450 degC initial temperatures.

Uses the original timing histories and scalar LHS with a new scalar seed.
Run without --prepare-only to execute, export and write the verified report.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from esatan_launcher import Install
from pcb_v3.generation import variable
from pcb_v3.generation.design import PARAMETERS
from pcb_v3.esatan import model
from esatan_data.collection import SolvedSweep
from pcb_v3.workflow import initialize, locked, run, save_json

BOUNDS = ((100., 450.), (10., 50.), (10., 30.))


def audit(deck):
    text = Path(deck).read_text()
    nodes = {int(n): {'type': kind, 'label': label} for kind, n, label in
             re.findall(r"(?m)^\s*([DBAX])(\d+)\s*=\s*'([^']*)'", text)}
    initial = {int(n): value.strip() for n, value in
               re.findall(r'(?m)^\s*T(\d+)\s*=\s*([^\n]+)', text)}
    free = sorted(n for n, info in nodes.items() if info['type'] in ('D', 'A') and n != 268)
    if not free or any(initial.get(n) != 'T_enviroment' for n in free):
        raise ValueError('hypothesis false: free-node initialization differs')
    if initial.get(268) != 'T_Interface' or not re.search(r"CALL\s+STATST\('N268','B'\)", text):
        raise ValueError('unexpected interface boundary')
    uses = []
    for number, line in enumerate(text.splitlines(), 1):
        if not re.search(r'\bT_enviroment\b', line, re.I):
            continue
        uses.append({'line': number, 'text': line.strip()})
        if not (re.match(r'\s*T\d+\s*=\s*T_enviroment\s*$', line)
                or re.match(r'\s*T_enviroment\s*=', line)
                or re.match(r"\s*B99999\s*=\s*'ENVIRONMENT',\s*T\s*=\s*T_enviroment,", line)):
            raise ValueError(f'hypothesis false: additional parameter use on line {number}')
    edges = re.findall(r'(?im)^\s*(GL|GR|GF|GV)\s*\(([^)]*)\)', text)
    links = [(kind, args) for kind, args in edges if re.search(r'\b99999\b', args)]
    if links:
        raise ValueError(f'hypothesis false: ENVIRONMENT has links: {links}')
    # input_paths also rejects INCLUDE/GLOBAL dependencies.
    model.input_paths(deck)
    return {'confirmed': True, 'source': str(Path(deck).resolve()),
            'source_sha256': model.digest(deck), 'free_node_ids': free,
            'free_nodes_initialized_to_environment': len(free),
            'exceptions': {'268': 'prescribed T_Interface', '99998': 'inactive, initially 0 degC'},
            'environment_node': 99999, 'environment_links': links,
            'parameter_uses': uses,
            'interpretation': 'T_enviroment sets initial temperatures and the isolated ENVIRONMENT node; it is not a coupled ambient boundary.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--source-deck', type=Path, default=model.DEFAULT_DECK)
    p.add_argument('--install-root', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    evidence = audit(args.source_deck)
    config = variable.Config(total_samples=500, always_on=25, always_off=25,
                             scalar_seed=20261104, timing_seed=42, scalar_bounds=BOUNDS)
    work = args.work.expanduser().resolve()
    with locked(work):
        trials, deck = initialize(work, config, args.source_deck)
        save_json(work / 'deck-audit.json', evidence)
        save_json(work / 'generation-settings.json', {
            'scalar_seed': config.scalar_seed,
            'timing_seed': config.timing_seed, 'cases': len(trials),
            'bounds': dict(zip(PARAMETERS, BOUNDS)),
            'horizon_seconds': 2200, 'native_step_seconds': 5, 'export_dtype': 'float32'})
    print(f'Prepared {len(trials)} cases: {work}', flush=True)
    if args.prepare_only:
        return
    output = run(work, config, Install(args.install_root), args.source_deck)
    sweep = SolvedSweep(work / 'sweep', deck=deck)
    report = {'dataset': str(output), 'cases': [], 'initial_temperatures_verified': True}
    free_ids = np.asarray(evidence['free_node_ids'])
    for i, ref, case in sweep.iter_cases():
        trial = trials[i]
        physical = np.isin(case.node_numbers, np.r_[free_ids, 268])
        free = np.isin(case.node_numbers, free_ids)
        temperatures = case.temperature
        if case.get_temperature_unit() == 'Kelvin':
            temperatures = temperatures - 273.15
        residual = np.abs(temperatures[free, 0] - trial.scalars['T_enviroment'])
        if len(residual) != len(free_ids) or residual.max() > 1e-3:
            raise ValueError(f'{ref.id}: initial free-node temperatures do not match the sampled parameter')
        report['cases'].append({'id': ref.id, **trial.scalars,
            'initial_max_abs_error_degC': float(residual.max()),
            'pcb_free_min_degC': float(temperatures[free].min()),
            'pcb_free_max_degC': float(temperatures[free].max()),
            'physical_min_degC': float(temperatures[physical].min()),
            'physical_max_degC': float(temperatures[physical].max())})
    for key in ('pcb_free_min_degC', 'physical_min_degC'):
        report[key] = min(row[key] for row in report['cases'])
    for key in ('pcb_free_max_degC', 'physical_max_degC', 'initial_max_abs_error_degC'):
        report[key] = max(row[key] for row in report['cases'])
    report['sampled_ranges'] = {name: [min(t.scalars[name] for t in trials), max(t.scalars[name] for t in trials)] for name in PARAMETERS}
    save_json(work / 'temperature-report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'cases'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
