"""Discover exact repeated axes without assigning training roles."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Field:
    axes: tuple[str, ...]
    shape: tuple[int, ...]
    has_nan: bool


def case_fields(case, stride):
    """Yield (HDF5 path, time-by-entity array, entity axis)."""
    for name in case.node_feature_names:
        if not name or "/" in name:
            raise ValueError(f"Invalid field name: {name!r}")
        yield f"fields/{name}", case.attr(name)[:, ::stride].T, "node"
    for kind in case.conductor_types:
        yield f"conductors/{kind}/values", case.conductor_series(kind)[::stride], "edge"


def inspect_fields(sweep, stride=1):
    """Find stored axes with memory bounded by a few cases.

    Compare source precision before dtype conversion. NaNs compare equal only
    at the same places. Keep edge axes to align values with connectivity.
    """
    if (
        isinstance(stride, bool)
        or not isinstance(stride, (int, np.integer))
        or stride < 1
    ):
        raise ValueError("stride must be a positive integer")
    first, varying, has_nan, entities = {}, {}, {}, {}
    for i, ref, case in sweep.iter_cases():
        for path, block, entity in case_fields(case, stride):
            if np.isinf(block).any():
                raise ValueError(f"{ref.id}: infinite values in {path}")
            if i == 0:
                first[path] = block.copy()
                varying[path] = [False, False, entity == "edge"]
                has_nan[path] = False
                entities[path] = entity
            flags = varying[path]
            flags[0] |= not np.array_equal(block, first[path], equal_nan=True)
            flags[1] |= not np.array_equal(
                block, np.broadcast_to(block[:1], block.shape), equal_nan=True
            )
            flags[2] |= not np.array_equal(
                block, np.broadcast_to(block[:, :1], block.shape), equal_nan=True
            )
            has_nan[path] |= bool(np.isnan(block).any())
    result = {}
    for path, block in first.items():
        axes = ("case", "time", entities[path])
        shape = (sweep.n_cases, *block.shape)
        keep = varying[path]
        result[path] = Field(
            tuple(axis for axis, flag in zip(axes, keep) if flag),
            tuple(size for size, flag in zip(shape, keep) if flag),
            has_nan[path],
        )
    return result
