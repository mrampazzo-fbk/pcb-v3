from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from esatan_results_reader.esatan_tmd import ThermalResultError

from esatan_data.case import TmdCase
from esatan_data.writer import export_fields
from functools import partial
export_fields = partial(export_fields, parameter_units={"power":"W"})
from esatan_data.collection import SolvedSweep


class Reader:
    unit_map={'temperature':'degC'}
    attr_map={'Temperature':'temperature'}
    def get_conductor_types(self): return ('GL',)
    def get_node_types(self): return np.array(['D'] * len(self.get_nodes()))
    def get_field_units(self):
        return {name: {'Temperature':'degC','Capacitance':'J/K','Total_Internal_Heat_Source':'W'}[name]
                for name in self.get_nodal_attribute_names()}
    def get_conductor_units(self): return {'GL':'W/K'}
    def get_nodes(self):return [{'number':250,'label':'CPU','full_path':'PCB'},{'number':268,'label':'sink','full_path':'PCB'}]
    def get_times(self):return [0.,5.,10.]
    def get_t_abs(self):return 273.15
    def get_nodal_attribute_names(self):return ['Temperature','Capacitance','Total_Internal_Heat_Source']
    def get_nodal_attribute_data(self):return np.array([[[20.,2.,0.],[30.,3.,0.]],[[21.,2.,1.],[30.,3.,0.]],[[22.,2.,0.],[30.,3.,0.]]])
    def get_conductors(self,conductor_type):
        if conductor_type[0]!='GL':raise ThermalResultError('absent')
        return {'GL':{'node_indexes':np.array([[0,1,1,1]]),'values':np.array([[[1.]],[[2.]],[[3.]]])}}


@pytest.fixture
def sweep(tmp_path):
    sweep=object.__new__(SolvedSweep)
    sweep._validate_science = None
    sweep.cases=(SimpleNamespace(id='a'),SimpleNamespace(id='b'))
    sweep.params=np.array([[10.],[20.]])
    sweep.param_names=['power']
    cases=[TmdCase(Reader()),TmdCase(Reader())]
    cases[1].node_features[0]+=1
    sweep.case=lambda i:cases[i]
    return sweep


def test_repeated_axes_are_omitted_without_losing_values(tmp_path,sweep):
    output=export_fields(sweep,tmp_path/'out.h5',boundary_nodes=[268])
    with h5py.File(output) as data:
        assert data.attrs['format'] == 'esatan-data'
        assert 'dataset_kind' not in data.attrs
        assert 'provenance' not in data
        assert 'model_mask' not in data['nodes']
        assert data['fields/Temperature'].shape==(2,3,2)
        assert data['fields/Temperature'].compression=='gzip'
        assert tuple(data['conductors/GL/values'].attrs['axes'])==('time','edge')
        assert data['conductors/GL/values'].shape==(3,1)
        for field in sweep.case(0).node_feature_names:
            stored=data[f'fields/{field}']
            axes=tuple(stored.attrs['axes'])
            shape=tuple(stored.shape[axes.index(axis)] if axis in axes else 1
                        for axis in ('case','time','node'))
            actual=np.broadcast_to(stored[()].reshape(shape),(2,3,2))
            for i in range(2):
                expected=sweep.case(i).attr(field).T
                if field=='Temperature':expected=expected+273.15
                np.testing.assert_array_equal(actual[i],expected.astype('float32'))
                np.testing.assert_array_equal(data['conductors/GL/values'][:],sweep.case(i).conductor_series('GL'))


def test_failed_validation_leaves_no_partial_hdf5(tmp_path,sweep):
    sweep.case(1).times[-1]=15.
    with pytest.raises(ValueError,match='times differs'):
        export_fields(sweep,tmp_path/'bad.h5')
    assert not list(tmp_path.iterdir())


def test_existing_output_untouched(tmp_path,sweep):
    output=tmp_path/'keep.h5'
    output.write_bytes(b'keep')
    with pytest.raises(FileExistsError):export_fields(sweep,output)
    assert output.read_bytes()==b'keep'


def test_overflow_rejected_before_publication(tmp_path,sweep):
    sweep.case(0).node_features[0]=1e100
    with pytest.raises(ValueError,match='overflows'):
        export_fields(sweep,tmp_path/'bad.h5')
    assert not list(tmp_path.iterdir())
