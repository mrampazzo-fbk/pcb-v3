"""Generate PCB_v3 datasets: single, variable or catalog."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .generation import catalog, design, variable
from .esatan.model import DEFAULT_DECK, source_profile
from .generation.profiles import Profile
from .workflow import dataset_name, export, family, generator, load_design, run


def _catalog_profiles(args):
    profiles = catalog.PROFILES
    if args.catalog is not None:
        doc = json.loads(args.catalog.read_text())
        if doc['horizon_seconds'] != design.HORIZON:
            raise ValueError('catalog horizon must be 2200 seconds')
        profiles = tuple(Profile(p['id'], tuple(map(tuple, p['changes']))) for p in doc['scenarios'])
    if args.profiles is not None:
        lookup = {p.id: p for p in profiles}
        unknown = set(args.profiles) - lookup.keys()
        if unknown:
            raise ValueError(f'unknown catalog profiles: {sorted(unknown)}')
        profiles = tuple(lookup[name] for name in args.profiles)
    return profiles


def _configuration(args, saved=None):
    kind = args.configuration or (family(saved) if saved is not None else None)
    if kind is None:
        raise ValueError('choose a configuration: single, variable or catalog')
    if saved is not None and family(saved) != kind:
        raise ValueError('configuration differs from the saved design')
    names = ('samples', 'scalar_seed', 'timing_seed', 'min_on', 'min_off', 'intervals',
             'on_fraction', 'concentrations', 'always_on', 'always_off')
    options = {name: getattr(args, name) for name in names if getattr(args, name) is not None}
    if kind == 'variable':
        if 'samples' in options and (saved is None or getattr(saved, 'total_samples', None) is not None):
            options['total_samples'] = options.pop('samples')
        for name in ('intervals', 'on_fraction', 'concentrations'):
            if name in options:
                options[name] = tuple(options[name])
    if args.catalog is not None or args.profiles is not None:
        if kind != 'catalog':
            raise ValueError('--catalog and --profiles require the catalog configuration')
        options['profiles'] = _catalog_profiles(args)
    if saved is not None:
        unsupported = options.keys() - saved.__dataclass_fields__.keys()
        if unsupported or replace(saved, **options) != saved:
            raise ValueError('run options differ from the saved design')
        return saved
    defaults = {
        'single': design.Config(samples=500),
        'variable': variable.Config(total_samples=500, always_on=25, always_off=25),
        'catalog': catalog.Config(),
    }
    config = defaults[kind]
    unsupported = options.keys() - config.__dataclass_fields__.keys()
    if unsupported:
        raise ValueError(f'options not applicable to {kind}: {sorted(unsupported)}')
    config = replace(config, **options)
    if kind == 'single':
        points, period = source_profile(args.source_deck)
        config = replace(config, source_profile=points, period=period)
    return config


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='pcb-v3', description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('preview', 'run'):
        p = commands.add_parser(command)
        p.add_argument('configuration', choices=('single', 'variable', 'catalog'),
                       nargs='?' if command == 'run' else None)
        p.add_argument('--samples', '--total-samples', dest='samples', type=int,
                       help='total cases (default 500)')
        p.add_argument('--scalar-seed', type=int)
        p.add_argument('--timing-seed', type=int)
        p.add_argument('--min-on', type=float)
        p.add_argument('--min-off', type=float)
        p.add_argument('--intervals', type=int, nargs='+')
        p.add_argument('--on-fraction', type=float, nargs=2, metavar=('LOW', 'HIGH'))
        p.add_argument('--concentrations', type=float, nargs='+')
        p.add_argument('--always-on', type=int)
        p.add_argument('--always-off', type=int)
        p.add_argument('--profiles', nargs='+', help='select named catalog profiles')
        p.add_argument('--catalog', type=Path, help='custom binary timing catalog JSON')
        p.add_argument('--source-deck', type=Path, default=DEFAULT_DECK)
        if command == 'preview':
            p.add_argument('--out', type=Path, required=True)
        else:
            p.add_argument('--work', type=Path, required=True)
            p.add_argument('--install-root', type=Path, default=Path('/home/esatan/ESATAN-TMS'))
    p = commands.add_parser('export')
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--step', type=int, choices=(5, 50), default=5)
    args = parser.parse_args(argv)
    try:
        if args.command == 'export':
            work = args.work.resolve()
            config, _ = load_design(work / 'design.json')
            print(export(work, args.out or work / dataset_name(config, args.step), args.step))
            return
        saved = None
        if args.command == 'run' and (args.work / 'design.json').exists():
            saved, _ = load_design(args.work / 'design.json')
        config = _configuration(args, saved)
        module = generator(config)
        trials = module.make_design(config)
        if args.command == 'preview':
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open('x') as out:
                json.dump(module.document(config, trials), out, indent=2)
                out.write('\n')
            print(args.out)
        else:
            from esatan_launcher import Install
            print(run(args.work, config, Install(args.install_root), args.source_deck))
        print(f'{family(config)}: {len(trials)} cases; array capacity {len(trials[0].points)}')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
