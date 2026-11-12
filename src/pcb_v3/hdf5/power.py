"""PCB export recipe: native power, interval forcing and initial properties."""
import numpy as np

from esatan_data.fields import case_fields
from esatan_data.writer import STRINGS

POWER = 'Total_Internal_Heat_Source'
INSTANTANEOUS = POWER + '_Instantaneous'


def export_blocks(trials, node_weights, step, simulation):
    """Transform source fields before automatic axis inspection and casting."""
    by_id = {trial.id: trial for trial in trials}

    def blocks(case, stride, ref):
        trial = by_id[ref.id]
        times = case.times[::stride]
        weights = np.array([node_weights.get(int(n), 0.) for n in case.node_numbers])
        fractions = trial.profile.interval_mean(times, step, simulation.horizon_seconds)
        mean = (trial.scalars['Q_Heat_Source'] * fractions)[:, None] * weights
        if not np.isfinite(mean).all():
            raise ValueError(f'{ref.id}: non-finite interval power')
        for path, values, entity in case_fields(case, stride):
            if path == f'fields/{POWER}':
                yield f'fields/{INSTANTANEOUS}', values, entity
                yield path, mean, entity
            elif path in ('fields/Capacitance', 'conductors/GL/values'):
                # This is a recipe decision, not automatic constancy detection.
                yield path, np.broadcast_to(values[:1], values.shape), entity
            else:
                yield path, values, entity
    return blocks


def field_metadata(step):
    return {
        f'fields/{POWER}': dict(units='W', sampling='forward_interval_mean',
                               interval_seconds=step,
                               final_sample='zero; no interval beyond horizon'),
        f'fields/{INSTANTANEOUS}': dict(units='W', sampling='instantaneous'),
        'fields/Capacitance': dict(sampling='initial_value', source_time_seconds=0.),
        'conductors/GL/values': dict(sampling='initial_value', source_time_seconds=0.),
    }


def add_total_power(data):
    """Sum the published nodal interval watts in float64, one case at a time."""
    field = data[f'fields/{POWER}']
    axes = tuple(field.attrs['axes'])
    count, times, nodes = len(data['cases/ids']), len(data['time']), len(data['nodes/ids'])
    total = data.create_dataset('forcing/Q', shape=(count, times), dtype='float64',
                               chunks=(1, times), compression='gzip', shuffle=True)
    local_axes = tuple(a for a in axes if a != 'case')
    for i in range(count):
        values = field[i] if 'case' in axes else field[()]
        shape = tuple(values.shape[local_axes.index(a)] if a in local_axes else 1
                      for a in ('time', 'node'))
        total[i] = np.broadcast_to(values.reshape(shape), (times, nodes)).sum(axis=1, dtype='float64')
    total.attrs.update(dict(field.attrs))
    total.attrs['axes'] = np.asarray(('case', 'time'), dtype=STRINGS)
