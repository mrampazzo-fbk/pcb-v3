# pcb-v3

Generate PCB thermal datasets with ESATAN-TMS and export them to HDF5.

## Installation

Requires Linux, Python ≥ 3.11, uv, `esatan-launcher` ≥ 0.4, and `esatan-results-reader`.
Simulations also require a licensed ESATAN-TMS installation and the PCB source
model (`.d`, `.acd`, `Template.tpl`).

This development branch uses the sibling checkout `../esatan-launcher` in editable
mode. Both repositories must contain the `refactor/dataset-responsibilities`
changes. The relative source in `pyproject.toml` and `uv.lock` is intentional until
the launcher changes are published; it should then be replaced with a pinned Git
revision containing the new API.

```bash
uv sync --locked --no-dev
uv run pcb-v3 --help
```

## Usage

Available configurations: `single` (the model's periodic profile),
`variable` (variable pulses), and `catalog` (predefined profiles).
Each configuration generates 500 cases by default; use `--samples` to change
the total.

```bash
# Preview cases without running the solver
uv run pcb-v3 preview variable --out runs/preview.json

# Run simulations and export HDF5 at 5 s intervals
uv run pcb-v3 run variable --work runs/variable \
  --source-deck /path/to/model/PCB.d \
  --install-root /path/to/ESATAN-TMS

# Resume the same study
uv run pcb-v3 run --work runs/variable \
  --source-deck /path/to/model/PCB.d \
  --install-root /path/to/ESATAN-TMS

# Export completed results at 50 s intervals
uv run pcb-v3 export --work runs/variable --step 50
```

Use a new working directory for each study. Results and case definitions are
stored in `--work`; existing HDF5 files are never overwritten.
For command options, run `uv run pcb-v3 <command> --help`.

Without `--install-root`, the launcher uses `ESATAN_TMS_DIR` or its remembered
installation. The default simulation lasts 2200 seconds with a native step of
5 seconds. `--horizon-seconds` and `--native-step-seconds` set these explicitly;
export steps must be multiples of the native step and divide the horizon.
Catalog event times are absolute and are not stretched when the horizon changes.

## Python API and responsibilities

```python
from pathlib import Path
from esatan_launcher import Install
from pcb_v3 import VariableConfig, run, export

work = Path("runs/variable")
config = VariableConfig(total_samples=500, always_on=25, always_off=25)
output = run(work, config, Install(), source=Path("/path/to/model/PCB.d"))
coarse = export(work, step=50)
```

`SingleConfig`, `VariableConfig` and `CatalogConfig` carry the simulation time
settings as well as the design settings. For `SingleConfig`, supply
`source_profile` and `period` from `pcb_v3.esatan.model.source_profile(deck)`.
`generation.<family>.make_design(config)` previews trials without execution;
`load_design(path)` reads and verifies a saved design.

The package owns the PCB experiment: DOE, profiles, model preparation, physical
validation and scientific dataset transformations. `esatan-data-collection` owns
automatic axis inspection and the shared HDF5 writer. `esatan-launcher` owns PAR rendering,
processes, attempts, case artifacts and tracker transitions. `pcb-v3` supplies a
result validator to the launcher when resuming; it never edits tracker records.
`esatan-results-reader` owns TMD parsing. The conversion from a trial to ESATAN
inputs lives in `esatan/inputs.py`, outside the DOE.

`run` and `export` validate the design, model bundle, TMD time grid, loaded and
boundary nodes, initial free-node temperatures, and nodal power. The complete
export retains `format="pcb-v3"` and schema version 2: temperatures are in kelvin,
`fields/Total_Internal_Heat_Source` is the forward interval mean (the training
default), with zero at the final sample because no interval follows the horizon.
`fields/Total_Internal_Heat_Source_Instantaneous` preserves the native TMD values,
including the solver's convention at discontinuities. `forcing/Q` sums the
published interval nodal watts in float64. Interval means are recomputed from
profiles for each export resolution; instantaneous power is only subsampled.

Capacitance and GL use their first native frame for each case, recorded as
`sampling="initial_value"`. This is an export decision: it does not claim that
the original simulation used time-independent properties. GR and other fields
retain their source evolution. All fields undergo exact automatic axis inspection
before casting: no full `(case,time,node)` storage is imposed. No new interval
axis is introduced, and schema version 2 is retained.

Exports record SHA-256 references to `design.json` and `inputs/bundle.json`.
Keep these files with the HDF5; detailed timing events remain in the design.
Existing exports are never overwritten or automatically converted.

The shared writer is `esatan_data.writer.export_fields`. Its default exports
native fields with `format="esatan-data"`; PCB supplies transformed blocks before
inspection, parameter units and scientific metadata. Publication remains atomic,
including failures while writing derived power. Install the sibling collector
with `uv sync`; it contains the sole writer used by PCB and CubeSat export.

## Study format

Studies use one format, `pcb-study-v1`, with an explicit generator, every
configuration field (including defaults), and the generated cases. Persistence
lives in `generation/study.py`; the scientific generators only create trials.
Model bundles record the complete time settings and preparation recipe 2.

There are no converters for previous study formats, model bundles or tracker
hashes. Regenerate studies in new work directories. Resume is supported for runs
created with the current format.

The long-trajectory script uses the common workflow and writes `design.json`
and `pcb-single-profile-{5,50}s.h5` in a new work directory.
`--prepare-only` exercises model preparation without requiring an installation or
a licence.

## Initial-temperature extension

To generate the 500-case study with the same variable timing histories,
initial free-node temperatures sampled by LHS over 100–450 °C, and unchanged
interface/power ranges:

```bash
.venv/bin/python scripts/pcb_temperature_extension.py --work runs/temperature-new
```

The script audits the source deck before invoking ESATAN and uses scalar seed
20261104, timing seed 42, 450 variable cases, 25 always-on cases and 25 always-off
cases. It regenerates the design directly without reading an old study and
stores the complete configuration and bounds in `--work`. It generates the same
5 s, 2200 s, float32 HDF5 format as the baseline. `--prepare-only` writes and
validates inputs without using a licence. On successful completion,
`temperature-report.json` records per-case and global temperature extrema in °C
and verifies all 290 free nodes' initial temperatures against their sampled
`T_enviroment` (tolerance 0.001 °C). Report extrema explicitly distinguish free
PCB nodes from the physical model including prescribed interface node 268;
ENVIRONMENT and the inactive node are excluded from physical extrema.
