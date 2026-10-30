"""Read or replace one ACD array while preserving all other model data."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np

ARRAY_NAME = "bc_Profilo_Clic001"


def _array(group, name):
    records = group["ArrayInformation"][:]
    names = [row["arrayName"].decode().rstrip("\x00") for row in records]
    if names.count(name) != 1:
        raise ValueError(f"ACD must contain exactly one {name} array")
    index = names.index(name)
    row = records[index]
    if int(row["dimension1"]) != 2 or int(row["size"]) != 2 * int(row["dimension2"]):
        raise ValueError(f"{name}: expected a two-column time/value array")
    offset = sum(int(row["size"]) for row in records[:index])
    return index, offset, records


def read_profile_points(path: Path, array_name: str = ARRAY_NAME) -> np.ndarray:
    with h5py.File(path, "r") as data:
        group = data["ModelData/Arrays"]
        index, offset, records = _array(group, array_name)
        size = int(records[index]["size"])
        return group["ArrayData"][offset:offset + size].reshape(-1, 2)


def generate_acd(source: Path, destination: Path, points, *, array_name: str = ARRAY_NAME) -> Path:
    """Copy an ACD and replace a profile, shifting subsequent arrays as needed.

    The source supplies material tables and model metadata; an ACD cannot be
    generated from a timing history alone. Publish only a complete new file.
    """
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
            or not np.isfinite(points).all() or points[0, 0] != 0
            or np.any(np.diff(points[:, 0]) <= 0) or np.any(points[:, 1] < 0)):
        raise ValueError("profile points must be finite, increasing from zero, with nonnegative values")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".acd-", dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        shutil.copy2(source, staged)
        with h5py.File(staged, "r+") as data:
            group = data["ModelData/Arrays"]
            index, offset, records = _array(group, array_name)
            old_size = int(records[index]["size"])
            used = int(group.attrs["numberOfArrayValues"])
            if used != sum(int(row["size"]) for row in records):
                raise ValueError("ACD array sizes disagree with numberOfArrayValues")
            values = group["ArrayData"]
            replacement = np.concatenate((values[:offset], points.ravel(), values[offset + old_size:used]))
            if len(replacement) > len(values):
                values.resize((len(replacement),))
            values[:len(replacement)] = replacement
            if len(replacement) < used:
                values[len(replacement):used] = 0
            records[index]["size"] = points.size
            records[index]["dimension2"] = len(points)
            group["ArrayInformation"][index] = records[index]
            group.attrs.modify("numberOfArrayValues", len(replacement))
        os.link(staged, destination)
    return destination
