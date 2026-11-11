"""One immutable deck/ACD bundle; source files are only read."""
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from esatan_launcher.deck import parse_deck

from ..generation.design import CAPACITY
from ..time_grid import Simulation
from .acd import ARRAY_NAME, generate_acd, read_profile_points

BOUNDARY_NODES = (268, 99998, 99999)
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[dDeE][+-]?\d+)?"
LOAD = re.compile(
    rf"(?m)^(?P<indent>[ \t]*)QI(?P<node>\d+)\s*=\s*(?P<weight>{NUMBER})\s*\*\s*"
    rf"(?:(?:Q_Scale|Q_Heat_Source)\s*\*\s*)?"
    rf"INTCY1\(\s*TIMEM\s*,\s*{ARRAY_NAME}\s*,\s*0\s*,\s*"
    rf"(?P<period>{NUMBER})\s*,\s*0(?:\.0)?(?:D0)?\s*\)[ \t]*$",
    re.IGNORECASE,
)


DEFAULT_DECK = Path('/home/esatan/PCB_Model_Task1/PCB_v3/esatan/Tansient_Q_Source_Cycle/PCB_Clamp_CPUs_Box_Param_Tansient_Q_Source_Cycle.d')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_profile(source):
    """Return the source ACD values and the common INTCY1 period."""
    deck, acd, _ = input_paths(source)
    periods = {float(m['period'].upper().replace('D', 'E')) for m in LOAD.finditer(deck.read_text())}
    if len(periods) != 1 or sorted(weights(deck)) != list(range(250, 268)):
        raise ValueError('expected eighteen CPU loads with one profile period')
    return tuple(map(tuple, read_profile_points(acd).tolist())), periods.pop()


def prepare(source, directory, points, *, period=None, simulation=Simulation()):
    source, acd, template = input_paths(source)
    directory = Path(directory)
    request = {'sources': {p.name: digest(p) for p in (source, acd, template)},
               'points': np.asarray(points).tolist(), 'recipe': 2,
               'period': simulation.horizon_seconds + 1 if period is None else period,
               'simulation': {'horizon_seconds': simulation.horizon_seconds,
                              'native_step_seconds': simulation.native_step_seconds}}
    if directory.exists():
        manifest = json.loads((directory / 'bundle.json').read_text())
        if manifest['request'] != request or any(digest(directory / n) != h for n, h in manifest['files'].items()):
            raise ValueError('prepared model changed; use a new work directory')
        return directory / source.name
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix='.model-') as tmp:
        staged = Path(tmp) / 'inputs'
        staged.mkdir()
        deck = staged / source.name
        capacity = len(points)
        text = rewrite_deck(source, capacity=capacity, simulation=simulation,
                            period=period if period is not None else simulation.horizon_seconds + 1)
        text = re.sub(rf'(?m)^([ \t]*OUTINT\s*=\s*){NUMBER}',
                      rf'\g<1>{float(simulation.native_step_seconds)}', text)
        deck.write_text(text)
        generate_acd(acd, staged / f'{parse_deck(source).model_name}.acd', points)
        shutil.copy2(template, staged / 'Template.tpl')
        manifest = {'request': request, 'files': {p.name: digest(p) for p in staged.iterdir()}}
        (staged / 'bundle.json').write_text(json.dumps(manifest, indent=2) + '\n')
        staged.rename(directory)
    return directory / source.name


def weights(deck):
    return {int(m['node']): float(m['weight'].upper().replace('D','E')) for m in LOAD.finditer(Path(deck).read_text())}



def input_paths(source):
    source = Path(source).expanduser().resolve()
    info = parse_deck(source)
    if not info.has_acdfile or not info.has_dmptmd or info.includes or info.global_file:
        raise ValueError('expected a self-contained PCB_v3 ACD/DMPTMD deck')
    return source, source.parent / f'{info.model_name}.acd', source.parent / 'Template.tpl'


def rewrite_deck(source, *, capacity=CAPACITY, period=None, simulation=Simulation()):
    period = simulation.horizon_seconds + 1 if period is None else period
    text = Path(source).read_text()
    if sorted(weights(source)) != list(range(250,268)):
        raise ValueError('expected the eighteen CPU loads QI250–QI267')
    text = re.sub(r'(?m)^[ \t]*Q_Scale\s*=.*\n', '', text)
    text, count = re.subn(rf'({ARRAY_NAME}\(\s*2\s*,\s*)\d+(\s*\)\s*/\s*DEFERRED\s*/;)',
                          rf'\g<1>{capacity}\2', text)
    if count != 1:
        raise ValueError('expected one deferred timing array')
    text = re.sub(rf'(?m)^([ \t]*TIMEND\s*=\s*){NUMBER}', rf'\g<1>{simulation.horizon_seconds:g}', text)
    return LOAD.sub(lambda m: f'{m["indent"]}QI{m["node"]} = {m["weight"]} * Q_Heat_Source * '
                    f'INTCY1(TIMEM, {ARRAY_NAME}, 0, {period:g}D0, 0.0D0)', text)


def prepared_deck(directory):
    """Verify the frozen model before reading or exporting its results."""
    directory = Path(directory)
    manifest = json.loads((directory / 'bundle.json').read_text())
    if manifest['request']['recipe'] != 2:
        raise ValueError('unsupported model bundle; regenerate in a new work directory')
    files = manifest['files']
    if any(Path(name).name != name or digest(directory / name) != value
           for name, value in files.items()):
        raise ValueError('prepared model changed; use a new work directory')
    decks = [directory / name for name in files if name.endswith('.d')]
    if len(decks) != 1:
        raise ValueError('prepared model must contain one deck')
    return decks[0]
