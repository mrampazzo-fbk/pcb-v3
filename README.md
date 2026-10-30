# pcb-v3

Generate PCB thermal datasets with ESATAN-TMS and export them to HDF5.

## Installation

Requires Linux, Python ≥ 3.11, uv, `esatan-launcher`, and `esatan-results-reader`.
Simulations also require a licensed ESATAN-TMS installation and the PCB source
model (`.d`, `.acd`, `Template.tpl`).

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
