# Multi-run statistics input

Drop the output of each run you want to compare into its **own sub-folder** here.
One sub-folder = one run. The folder name is used as the run label in the plots.

```
data/multi_run/
├── run_01/
│   └── checkpoint_1100.pkl      <- latest checkpoint of the run
├── run_02/
│   └── checkpoint_1100.pkl
├── run_03/
│   └── checkpoint_1100.pkl
...
```

## What file to put in each sub-folder

Only the **`stats` list** (one dict per iteration, produced by `QDRunner.run`)
is needed — the same data `ArchiveVisualizer.plot_stats` /
`plot_finetuning_val_*` read. That list is stored inside every checkpoint, so
just copy in the run's **latest `checkpoint_XXXX.pkl`** (highest iteration
number). If a folder has several `checkpoint_*.pkl` files, the one with the
highest iteration is used automatically.

The loader also accepts, in order of preference:

1. `checkpoint_*.pkl`  – full run checkpoint (scheduler is stubbed out on load;
   only the `stats` list is read, so you do **not** need the QD run
   dependencies / API server available).
2. `stats.pkl`         – a bare pickled `stats` list.
3. `stats.json`        – a JSON dump of the `stats` list.

You may also drop loose `*.pkl` / `stats.json` files directly in
`data/multi_run/` (each counts as one run, labelled by its filename) instead of
using sub-folders.

## Generating the plots

From the repo root:

```
python -m qd.multi_run_visualizer
```

Options:

```
python -m qd.multi_run_visualizer --runs-dir data/multi_run --out data/multi_run/_plots --smooth 11
```

Outputs (written to `--out`, default `data/multi_run/_plots`):

- `multi_run_stats.png`       – all per-iteration panels in one grid.
- `panels/<metric>.png`       – each panel as a standalone image.
- `multi_run_final_metrics.csv` – final-iteration value of each metric per run.
- `finetuning_val_loss_normalized.png` / `finetuning_val_kld_normalized.png`
- `finetuning_summary.png`
```

Every per-iteration panel overlays each run (thin coloured line) plus the
across-runs **mean** (bold black) and a **±1 std** band, aligned on a shared
iteration axis.

### Fine-tuning curves

Per-epoch fine-tuning curves can't be laid back-to-back across runs like they
are within a single run — early stopping gives each cycle a different epoch
count in every run. They are instead aligned on the **fine-tuning cycle** (the
QD iteration it ran at, which is shared across runs) and aggregated two ways:

- **`finetuning_*_normalized.png`** – each cycle's epoch axis is rescaled to
  progress 0→1 (0 = first epoch, 1 = saved/best epoch), interpolated onto a
  common grid, and averaged across runs. Cycles are coloured by QD iteration,
  with a ±1 std band, so the average convergence shape and its drift over the
  run are both visible.
- **`finetuning_summary.png`** – per cycle, the mean ±1 std of the first-epoch
  and best-epoch validation loss (their gap = how much the fine-tune helped)
  and the number of epochs to the saved model, plotted vs QD iteration.

These are skipped automatically for runs made with fine-tuning disabled.
