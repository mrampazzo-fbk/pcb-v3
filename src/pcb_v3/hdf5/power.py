"""Interval-mean nodal and total power for the HDF5 writer."""
import numpy as np

from ..time_grid import Simulation


def add_forcing(data, trials, node_weights, step, simulation=Simulation()):
    """Store interval-mean nodal watts and their total; keep profiles in the design."""
    weights = np.array([node_weights.get(int(n), 0.) for n in data['nodes/ids'][:]])
    amplitudes = np.array([t.scalars['Q_Heat_Source'] for t in trials])
    write_mean_power(data, [t.profile for t in trials], amplitudes, weights, step, simulation.horizon_seconds)


def write_mean_power(data, profiles, amplitudes, weights, step, horizon):
    """Integrate binary events before discarding them; Q sums stored nodal watts."""
    from .writer import STRINGS

    times = data['time'][:]
    amplitudes, weights = np.asarray(amplitudes), np.asarray(weights)
    if (amplitudes.shape != (len(profiles),) or weights.shape != (len(data['nodes/ids']),)
            or not np.isfinite(amplitudes).all() or not np.isfinite(weights).all() or step <= 0):
        raise ValueError('invalid power amplitudes, node weights or interval duration')
    for path in ('fields/Total_Internal_Heat_Source', 'forcing/Q'):
        if path in data:
            del data[path]
    field = data.create_dataset('fields/Total_Internal_Heat_Source',
        shape=(len(profiles), len(times), len(weights)), dtype='float32',
        chunks=(1, min(len(times), 128), len(weights)), compression='gzip', shuffle=True)
    total = data.create_dataset('forcing/Q', shape=(len(profiles), len(times)),
        dtype='float64', chunks=(1, len(times)), compression='gzip', shuffle=True)
    for i, profile in enumerate(profiles):
        fractions = profile.interval_mean(times, step, horizon)
        # Match the published files: 5 s averages their stored float32 on-loads;
        # 50 s averages nominal watts before distributing to nodes.
        if step == 5:
            on_power = np.asarray(amplitudes[i] * weights, dtype='float32')
            values = fractions[:, None] * on_power.astype('float64')
        else:
            values = (amplitudes[i] * fractions)[:, None] * weights
        block = np.asarray(values, dtype='float32')
        field[i] = block
        total[i] = block.sum(axis=1, dtype='float64')
    attributes = dict(units='W', sampling='forward_interval_mean', interval_seconds=step,
                      final_sample='zero; no interval beyond horizon', has_nan=False)
    field.attrs.update(axes=np.asarray(('case', 'time', 'node'), dtype=STRINGS), **attributes)
    total.attrs.update(axes=np.asarray(('case', 'time'), dtype=STRINGS), **attributes)
    data.attrs.update(horizon_seconds=horizon, output_step_seconds=step,
                      load_sampling='forward_interval_mean')
