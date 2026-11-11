"""One explicit study format; scientific generators do not handle persistence."""
from dataclasses import asdict
import json
from pathlib import Path

from . import catalog, design, variable
from .profiles import Profile

SCHEMA = 'pcb-study-v1'
GENERATORS = {'design': design, 'variable': variable, 'catalog': catalog}


def generator(config):
    for module in GENERATORS.values():
        if isinstance(config, module.Config):
            return module
    raise ValueError('unsupported study configuration')


def document(config, trials):
    module = generator(config)
    return {'schema': SCHEMA, 'generator': module.__name__.rsplit('.', 1)[-1],
            'config': asdict(config),
            'array_capacity': max(len(t.array_points) if t.array_points else getattr(t, 'capacity', design.CAPACITY)
                                  for t in trials),
            'cases': [asdict(t) for t in trials]}


def _tuples(value):
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


def load_design(path):
    doc = json.loads(Path(path).read_text())
    if doc.get('schema') != SCHEMA or doc.get('generator') not in GENERATORS:
        raise ValueError('unsupported study format; regenerate in a new work directory')
    module = GENERATORS[doc['generator']]
    settings = doc['config']
    if settings.keys() != module.Config.__dataclass_fields__.keys():
        raise ValueError('study configuration must specify every field')
    settings = {name: _tuples(value) for name, value in settings.items()}
    if module is catalog:
        settings['profiles'] = tuple(Profile(p['id'], _tuples(p['changes'])) for p in settings['profiles'])
    config = module.Config(**settings)
    trials = module.make_design(config)
    if doc != json.loads(json.dumps(document(config, trials))):
        raise ValueError('design differs from its seeds/configuration')
    return config, trials
