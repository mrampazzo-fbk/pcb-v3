"""Keep the launcher's CHANGE/hash/resume logic with a short CSV header.

The installed launcher automatically adds every CHANGE target to CSV_ENTITIES.
32 array references exceed ESATAN's 255-character header limit. Scope this small
adapter to one synchronous sweep; all array values remain in Case.values and in
the launcher's parameter-space hash. No installed dependency is modified.
"""
from threading import Lock
from unittest.mock import patch

import esatan_launcher.sweep as launcher

_LOCK = Lock()


class ArraySweep(launcher.Sweep):
    def render_par(self, cases=None):
        header = '$PARAMETERS, PARMONLY, CSV_ENTITIES = (T250, QI250)'
        selected = self.cases if cases is None else cases
        return header + '\n#\n' + '\n#\n'.join(c.render() for c in selected) + '\n'


def solve(install, deck, cases, directory):
    with _LOCK, patch.object(launcher, 'Sweep', ArraySweep):
        result = install.sweep(deck, cases, sweep_dir=directory, outputs=('T250', 'QI250'))
    if not result.ok:
        raise RuntimeError(f'incomplete sweep: {directory}; inspect tracker and solver logs')
    return result
