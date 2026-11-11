"""Named node attributes and complete conductor series from one solved TMD."""

from functools import cached_property

import numpy as np
from esatan_results_reader import EsatanTMD
from esatan_results_reader.esatan_tmd import ThermalResultError

from ..time_grid import Simulation

ESATAN_EDGE_TYPES = ("GL", "GR", "GF")

TEMPERATURE = "Temperature"
POWER = "Total_Internal_Heat_Source"


class TmdCase:
    """Node attributes `(attribute, node, time)` and complete conductor series."""

    def __init__(self, tmd: EsatanTMD):
        self.tmd = tmd
        self.nodes = self.tmd.get_nodes()
        self.node_features = np.transpose(tmd.get_nodal_attribute_data(), (2, 1, 0))
        self.node_feature_names = tmd.get_nodal_attribute_names()
        self._index = {n.lower(): i for i, n in enumerate(self.node_feature_names)}
        self._conductor_series = {}
        self.conductors = self._load_conductors()
        self.validate()

    @cached_property
    def node_numbers(self):
        """ESATAN node numbers, `(node,)`."""
        return np.array([int(n["number"]) for n in self.nodes])

    @cached_property
    def node_labels(self):
        """Node labels, `(node,)`."""
        return np.array([str(n["label"]) for n in self.nodes])

    @cached_property
    def times(self):
        """Solution times, `(n_timesteps,)`, read once per case, in seconds."""
        return np.asarray(self.tmd.get_times(), dtype=np.float64)

    def _attr_key(self, name):
        name = getattr(self.tmd, "reverse_attr_map", {}).get(name, name)
        return str(name).lower()

    def attr_index(self, name):
        """Look up a field by name or the reader's Attribute enum."""
        return self._index[self._attr_key(name)]


    def attr(self, name):
        """A nodal field with shape `(node, time)`."""
        return self.node_features[self.attr_index(name)]

    @property
    def temperature(self):
        return self.attr(TEMPERATURE)


    @property
    def power(self):
        return self.attr(POWER)


    def _load_conductors(self):
        """Keep complete conductor series and connectivity."""
        conductors = {}

        for c in ESATAN_EDGE_TYPES:
            try:
                cond = self.tmd.get_conductors(conductor_type=[c])[c]
            except ThermalResultError:
                # The upstream reader raises this when a conductor type is absent.
                continue

            node_indexes = cond["node_indexes"]
            submodel_cols = node_indexes[:, 2:4]
            if not np.all(submodel_cols == 1):
                raise ValueError(f"{c}: conductors spanning submodels are unsupported")

            cond_edges = node_indexes[:, :2]  # (n_edges, 2)
            series = cond["values"][:, :, 0]  # (n_timesteps, n_edges)
            self._conductor_series[c] = series

            conductors[f"{c}_edges"] = cond_edges

        return conductors

    @property
    def conductor_types(self):
        return tuple(sorted(self._conductor_series))


    def conductor_series(self, conductor_type):
        """Conductor values with shape `(time, edge)`."""
        return self._conductor_series[conductor_type]

    def validate(self):
        """Check this case's dimensions, time coordinates, and edge indices."""
        shape = (len(self.node_feature_names), len(self.nodes), len(self.times))
        if not all(shape) or self.node_features.shape != shape:
            raise ValueError(f"Invalid node data shape: expected {shape}")
        if len(set(self.node_numbers)) != len(self.nodes):
            raise ValueError("Node numbers must be unique")
        if len({name.lower() for name in self.node_feature_names}) != len(
            self.node_feature_names
        ):
            raise ValueError("Attribute names must be unique")
        if not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError("Time coordinates must be finite and strictly increasing")
        for kind, series in self._conductor_series.items():
            edges = self.conductors[f"{kind}_edges"]
            if edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu":
                raise ValueError(
                    f"{kind}: edges must be an integer array of node pairs"
                )
            if (edges < 0).any() or (edges >= len(self.nodes)).any():
                raise ValueError(f"{kind}: edge indices outside node range")
            if series.shape != (len(self.times), len(edges)):
                raise ValueError(f"{kind}: values must have shape (time, edge)")


    def get_temperature_unit(self):
        return self._temperature_unit

    @cached_property
    def _temperature_unit(self):
        attribute = self.tmd.attr_map.get(TEMPERATURE)
        return {"degC": "Celsius", "degK": "Kelvin"}.get(
            self.tmd.unit_map.get(attribute), "Unknown"
        )


def validate_solution(case, trial, node_weights, simulation=Simulation()):
    if not np.array_equal(case.times, simulation.times):
        raise ValueError(f'{trial.id}: incomplete or non-native TMD time grid')
    if not np.isfinite(case.temperature).all():
        raise ValueError(f'{trial.id}: invalid temperatures')
    missing = ({268, 99999} | set(node_weights)) - set(case.node_numbers)
    if missing:
        raise ValueError(f'{trial.id}: missing required nodes {sorted(missing)}')
    unit = case.get_temperature_unit()
    if unit not in ('Celsius', 'Kelvin'):
        raise ValueError(f'{trial.id}: unknown temperature units')
    temperatures = case.temperature - 273.15 if unit == 'Kelvin' else case.temperature
    for node, parameter in ((268, 'T_Interface'), (99999, 'T_enviroment')):
        observed = temperatures[case.node_numbers == node]
        if not np.allclose(observed, trial.scalars[parameter], rtol=0, atol=1e-3):
            raise ValueError(f'{trial.id}: boundary {parameter} differs')
    free = ~np.isin(case.node_numbers, (268, 99998, 99999))
    if not free.any() or not np.allclose(temperatures[free, 0], trial.scalars['T_enviroment'], rtol=0, atol=1e-3):
        raise ValueError(f'{trial.id}: initial free-node temperatures differ')
    # ESATAN may output the left limit at a discontinuity. Check every other
    # native sample, on every node, against its own original deck weight.
    check = ~np.any(np.isclose(case.times[:, None], np.array(trial.events)[:, 0], atol=1e-7, rtol=0), axis=1)
    expected = trial.scalars['Q_Heat_Source'] * trial.profile.power_at(case.times)
    weights = np.array([node_weights.get(int(n), 0.) for n in case.node_numbers])
    if not np.allclose(case.power[:, check], weights[:, None] * expected[check], rtol=1e-5, atol=1e-4):
        raise ValueError(f'{trial.id}: solver power differs from events/weights')
