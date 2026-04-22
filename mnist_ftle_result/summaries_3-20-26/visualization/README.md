# Visualization

Run:

```bash
python3 visualization/plot_research_results.py
```

Outputs go to `visualization/plots/`.

Current setup:

- `Plot 1`: mean `rho_all` vs depth with seed-wise standard deviation error bars, using `depth_sweep/summary_depth_sweep.csv`
- `Plot 2`: FTLE vs margin for `seed=0` at depths `4, 8, 16`
- `Plot 3`: FTLE and margin histograms for each depth
- `Plot 4`: mean `rho_all` vs width with seed-wise standard deviation error bars, using `width_sweep/summary.csv`
- `Plot 5`: FTLE vs margin for `seed=0` at widths `20, 50, 100` with `depth=4`
- `Plot 6`: FTLE and margin histograms for each width at `depth=4`

If `raw_npz/` exists in the workspace, the script will read:

- `raw_npz/*/artifacts/ftle_margin_data.npz`

By default, raw NPZ data is filtered to match the same hyperparameter sweep present in `depth_sweep/summary_depth_sweep.csv`, so extra runs such as other widths are ignored unless you opt in.

You can point the script at that data explicitly:

```bash
python3 visualization/plot_research_results.py --raw-data /path/to/raw_data_dir
```

To include all discovered widths and hyperparameter variants:

```bash
python3 visualization/plot_research_results.py --raw-data raw_npz --include-all-widths
```

Accepted raw formats:

- `.npz`
- `.csv`
- `.json`
- `.parquet`

Recognized column names:

- FTLE: `ftle`, `local_ftle`, `lyapunov`, `lambda`
- margin: `margin`, `logit_margin`, `signed_margin`, `classification_margin`
- depth: `depth`
- seed: `seed`
