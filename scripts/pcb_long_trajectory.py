#!/usr/bin/env python3
"""Generate one 22,000 s PCB-v3 reference using the existing suite.

Run from the repository with its environment, for example:
    .venv/bin/python scripts/pcb_long_trajectory.py --work runs/pcb-long

The source deck's periodic on/off profile is preserved. This script uses the
suite's launcher, validation and HDF5 writer. It requires the pcb-v3 Python
environment and a licensed ESATAN installation.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from esatan_launcher import Install
from pcb_v3 import SingleConfig, export, run
from pcb_v3.esatan import model
from pcb_v3.generation.design import PARAMETERS
from pcb_v3.workflow import initialize, locked

HORIZON = 22000.0


def random_profile(seed):
    """Ten independently generated 2200 s blocks, each with three pulses."""
    rng = np.random.default_rng(seed)
    events = []
    for block in range(10):
        durations = 30. + rng.dirichlet(np.ones(7)) * (2200. - 7 * 30.)
        starts = np.r_[0., np.cumsum(durations)[:-1]] + block * 2200.
        for index, time in enumerate(starts):
            level = float(index % 2)
            if not events or events[-1][1] != level:
                events.append((float(time), level))
    return tuple(events)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True, help='new study directory (or the same one to resume)')
    p.add_argument('--source-deck', type=Path, default=model.DEFAULT_DECK)
    p.add_argument('--install-root', type=Path)
    p.add_argument('--environment', type=float, default=25., help='initial free-node temperature in degrees C (default: 25)')
    p.add_argument('--interface', type=float, default=30., help='interface boundary in degrees C (default: 30)')
    p.add_argument('--power', type=float, default=20., help='heat source on-power in watts (default: 20)')
    p.add_argument('--profile', choices=('source', 'random'), default='source', help='original periodic profile or random pulses')
    p.add_argument('--seed', type=int, default=42, help='random profile seed (default: 42)')
    p.add_argument('--prepare-only', action='store_true', help='prepare inputs and design without launching ESATAN')
    args = p.parse_args()
    scalars = dict(zip(PARAMETERS, (args.environment, args.interface, args.power)))
    if not np.isfinite(list(scalars.values())).all() or args.power < 0:
        p.error('temperatures must be finite and power must be finite and nonnegative')
    work = args.work.expanduser().resolve()
    points, period = model.source_profile(args.source_deck)
    points = np.asarray(points, dtype=float)
    amplitude = points[:, 1].max()
    if amplitude <= 0 or not np.isin(points[:, 1], (0., amplitude)).all():
        p.error('the source profile must be binary with positive on level')
    normalized = tuple((float(t), float(v / amplitude)) for t, v in points)
    if args.profile == 'random':
        normalized = random_profile(args.seed)
        period = HORIZON + 1
    config = SingleConfig(samples=1, scalar_values=tuple(scalars.values()),
                          source_profile=normalized, period=period,
                          horizon_seconds=HORIZON)
    if args.prepare_only:
        with locked(work):
            _, deck = initialize(work, config, args.source_deck)
        print(f'Prepared {deck}; period={period:g} s; horizon={HORIZON:g} s', flush=True)
        return
    output = run(work, config, Install(args.install_root), args.source_deck)
    print(f'Exported {output}', flush=True)
    output_50 = work / 'pcb-single-profile-50s.h5'
    if not output_50.exists():
        print(f'Exported {export(work, output_50, step=50)}', flush=True)


if __name__ == '__main__':
    main()
