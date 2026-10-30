"""Export solved cases to HDF5, omitting exactly repeated axes."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from .fields import case_fields, inspect_fields

SCHEMA_VERSION = 2
FORMAT = "pcb-v3"
STRINGS = h5py.string_dtype("utf-8")
# These are the collection's output units; source temperature units are
# interpreted by TmdCase before conversion.
UNITS = {
    "Temperature": "K",
    "Capacitance": "J/K",
    "Area": "m^2",
    "Solar_Absorptivity": "1",
    "Infra-Red_Emissivity": "1",
    "X_Coordinate": "m",
    "Y_Coordinate": "m",
    "Z_Coordinate": "m",
}
CONDUCTOR_UNITS = {"GL": "W/K", "GR": "W/K^4"}


def _write(out, name, values, axes, **attrs):
    dataset = out.create_dataset(name, data=values)
    dataset.attrs.update(axes=np.asarray(axes, dtype=STRINGS), **attrs)


def _units(path):
    group, name = path.split("/", 1)
    if group == "conductors":
        return CONDUCTOR_UNITS.get(name.split("/")[0], "unknown")
    return "W" if "Heat_Source" in name else UNITS.get(name, "unknown")


def _boundary_mask(node_numbers, selected):
    selected = set(selected)
    unknown = selected - set(node_numbers)
    if unknown:
        raise ValueError(f"Unknown prescribed node numbers: {sorted(unknown)}")
    return np.isin(node_numbers, list(selected))


@contextmanager
def _new_hdf5(path):
    """Publish a complete file atomically, without replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        with h5py.File(temporary, "w") as out:
            yield out
        # Linking on the same filesystem also refuses a destination created
        # by another writer while this export was running.
        os.link(temporary, path)


def export_dataset(sweep, path, *, stride=1, dtype="float32", boundary_nodes=None, finalize=None):
    """Write a compact collection, publishing it only after a successful export.

    NaNs are preserved and temperatures converted to kelvin. None means boundary
    nodes are unknown; an empty sequence declares none. Existing files are refused.
    All nodes are retained; no learning-graph membership mask is written.
    Inspection and writing each stream through cases with bounded memory.
    """
    path = Path(path).expanduser()
    if path.exists():
        raise FileExistsError(path)
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")

    fields = inspect_fields(sweep, stride)
    first = sweep.first
    unit = first.get_temperature_unit()
    if "fields/Temperature" in fields and unit not in {"Celsius", "Kelvin"}:
        raise ValueError("Temperature units are unknown; cannot convert to kelvin")

    numbers = first.node_numbers
    boundary = (
        None
        if boundary_nodes is None
        else _boundary_mask(numbers, boundary_nodes)
    )

    with _new_hdf5(path) as out:
        out.attrs.update(
            format=FORMAT,
            schema_version=SCHEMA_VERSION,
            frame_stride=stride,
            boundary_source="unknown" if boundary_nodes is None else "explicit",
        )
        metadata = {
            "cases/ids": (
                np.asarray([c.id for c in sweep.cases], dtype=STRINGS), ("case",)
            ),
            "cases/parameters": (sweep.params, ("case", "parameter")),
            "cases/parameter_names": (
                np.asarray(sweep.param_names, dtype=STRINGS), ("parameter",)
            ),
            "nodes/ids": (numbers, ("node",)),
            "nodes/labels": (np.asarray(first.node_labels, dtype=STRINGS), ("node",)),
            "time": (sweep.times(stride), ("time",)),
        }
        if boundary is not None:
            metadata["nodes/boundary_mask"] = (boundary, ("node",))
        for name, (values, axes) in metadata.items():
            _write(out, name, values, axes)
        parameter_units = {'T_enviroment': 'degC', 'T_Interface': 'degC', 'Q_Heat_Source': 'W'}
        _write(out, 'cases/parameter_units',
               np.asarray([parameter_units.get(name, 'unknown') for name in sweep.param_names], dtype=STRINGS),
               ('parameter',))
        out["time"].attrs.update(units="s", sampling="instantaneous")

        out.create_group("fields")
        out.create_group("conductors")
        for kind in first.conductor_types:
            _write(
                out,
                f"conductors/{kind}/edges",
                first.conductors[f"{kind}_edges"],
                ("edge", "endpoint"),
                indexing="zero-based positions in nodes/ids",
            )
        for key, info in fields.items():
            options = (
                {"compression": "gzip", "shuffle": True}
                if info.shape and all(info.shape)
                else {}
            )
            dataset = out.create_dataset(key, shape=info.shape, dtype=dtype, **options)
            dataset.attrs.update(
                axes=np.asarray(info.axes, dtype=STRINGS),
                has_nan=info.has_nan,
                units=_units(key),
            )

        for i, ref, case in sweep.iter_cases():
            for key, block, entity in case_fields(case, stride):
                axes = fields[key].axes
                if i and "case" not in axes:
                    continue
                values = block[
                    slice(None) if "time" in axes else 0,
                    slice(None) if entity in axes else 0,
                ]
                if key == "fields/Temperature" and unit == "Celsius":
                    values = values + 273.15
                with np.errstate(over="ignore"):
                    values = np.asarray(values, dtype=dtype)
                if np.isinf(values).any():
                    raise ValueError(f"{ref.id}: {key} overflows {dtype}")
                out[key][i if "case" in axes else ()] = values
        if finalize is not None:
            finalize(out)
    return path
