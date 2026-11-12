from types import SimpleNamespace
import h5py
import numpy as np
import pytest
from pcb_v3.generation.design import make_design
from pcb_v3.generation.profiles import Profile
from pcb_v3.esatan.case import validate_solution
from pcb_v3.hdf5.power import export_blocks, field_metadata, add_total_power, POWER, INSTANTANEOUS
from pcb_v3.time_grid import Simulation
from esatan_data.writer import export_fields
from functools import partial
export_fields = partial(export_fields, parameter_units={"power":"W"})
from test_export import sweep


def expand(dataset, shape, axes=('case','time','node')):
    stored = tuple(dataset.attrs['axes'])
    sizes = tuple(dataset.shape[stored.index(a)] if a in stored else 1 for a in axes)
    return np.broadcast_to(dataset[()].reshape(sizes), shape)


def test_dual_power_initial_properties_and_automatic_axes(tmp_path, sweep):
    profile = Profile('pulse', ((0.,0.), (1.,1.), (4.,0.)))
    trials = [SimpleNamespace(id=ref.id, profile=profile, scalars={'Q_Heat_Source':20.}) for ref in sweep.cases]
    for i in range(2):
        sweep.case(i).node_features[1] += np.arange(3)[None,:] + i
    output = export_fields(sweep, tmp_path/'out.h5',
        blocks=export_blocks(trials, {250:.3,268:.7}, 5., Simulation(horizon_seconds=10)),
        field_attrs=field_metadata(5.), finalize=add_total_power)
    with h5py.File(output) as h5:
        mean = h5[f'fields/{POWER}']
        raw = h5[f'fields/{INSTANTANEOUS}']
        assert 'case' not in mean.attrs['axes']
        np.testing.assert_allclose(expand(mean,(2,3,2))[0], [[3.6,8.4],[0,0],[0,0]])
        np.testing.assert_array_equal(expand(raw,(2,3,2))[0], sweep.case(0).power.T)
        assert mean.attrs['sampling'] == 'forward_interval_mean'
        assert raw.attrs['sampling'] == 'instantaneous'
        assert 'time' not in h5['fields/Capacitance'].attrs['axes']
        assert 'case' in h5['fields/Capacitance'].attrs['axes']
        assert 'time' not in h5['conductors/GL/values'].attrs['axes']
        np.testing.assert_array_equal(h5['conductors/GL/values'][:], [1.])
        assert h5['forcing/Q'][0].sum()*5 == pytest.approx(60.)
        for i in range(2):
            np.testing.assert_array_equal(expand(h5['fields/Capacitance'], (2,3,2))[i,0], sweep.case(i).attr('Capacitance')[:,0])


def test_derived_power_overflow_leaves_no_file(tmp_path, sweep):
    trials = [SimpleNamespace(id=ref.id, profile=Profile('on',((0.,1.),)),
              scalars={'Q_Heat_Source':1e100}) for ref in sweep.cases]
    with pytest.raises(ValueError, match='overflows'):
        export_fields(sweep, tmp_path/'out.h5', field_attrs=field_metadata(5.), blocks=export_blocks(
            trials, {250:1.}, 5., Simulation(horizon_seconds=10)))
    assert not (tmp_path/'out.h5').exists()


def test_exact_interval_energy_including_partial_pulses():
    profile = Profile('pulse', ((0.,0.), (17.,1.), (83.,0.), (149.,1.), (211.,0.)))
    for step in (5.,50.):
        mean = profile.interval_mean(np.arange(0.,2201.,step), step, 2200.)
        assert mean[-1] == 0
        assert mean.sum()*step == pytest.approx(128.)


def fake_solution(trial):
    times = np.arange(0., 2201., 5.)
    q = trial.scalars['Q_Heat_Source'] * trial.profile.power_at(times)
    return SimpleNamespace(times=times, node_numbers=np.array([250,251,268,99999]),
        power=np.array([q*.3,q*.7,q*0,q*0]),
        temperature=np.array([q*0+trial.scalars['T_enviroment'],q*0+trial.scalars['T_enviroment'],q*0+trial.scalars['T_Interface'],q*0+trial.scalars['T_enviroment']]),
        get_temperature_unit=lambda: 'Celsius')


def test_checks_power_on_individual_nodes_not_only_total():
    trial = make_design()[250]
    case = fake_solution(trial)
    validate_solution(case,trial,{250:.3,251:.7})
    case.power[[0,1]] = case.power[[1,0]]
    with pytest.raises(ValueError, match='power'):
        validate_solution(case,trial,{250:.3,251:.7})


def test_incomplete_tmd_rejected():
    trial = make_design()[0]
    case = fake_solution(trial)
    case.times = case.times[:-1]
    with pytest.raises(ValueError,match='incomplete'):
        validate_solution(case, trial, {250:.3,251:.7})
