# Novelty Search for Racing Track Design

MSc thesis project to evolve diverse, high‑quality racing tracks with Quality‑Diversity search, run headless sims, and analyze with a data‑driven stack.

Based and forked from the previous work of [martinopiaggi](https://github.com/martinopiaggi), the previous version of the project can be found at: [Quality Diversity for Racing Track Design](https://github.com/martinopiaggi/Quality-Diversity-for-Racing-Track-Design) 

## What it is

- Quality‑Diversity search (Novelty Search with local competition over an unstructured archive) in a learned behavior space — no hand‑tuned features
- Behavior descriptors from a **VAE trained on driving telemetry**, finetuned online during the run ([AURORA](https://dl.acm.org/doi/abs/10.1145/3321707.3321804)‑style) so the latent space adapts to the tracks being discovered
- Voronoi and Convex Hull generators (genotype → spline → TORCS XML)
- Headless, containerized simulations with custom telemetry, parallelized with Dask
- Web visualizer for real‑time browse/debug
- Final output: **a track layout represented by a closed spline**, exported to TORCS‑compatible XML
- Procedural gen is very interesting. QD makes it diverse. Voronoi makes it weird in the best way.

## How it works (super concise)

- Genotype → Phenotype: 2D seed points → Voronoi/Convex Hull → closed spline → TORCS track XML
- Evaluate: headless races on Dask workers; telemetry (speed, curvature, steering, overtakes, incidents, closure error)
- Behavior space: a VAE encodes each track's telemetry sequence into a latent descriptor (32‑dim)
- Archive: pyribs `ProximityArchive` — a solution enters if it is novel enough (k‑NN distance over descriptors) or beats its nearest neighbors (local competition)
- Online retraining: every N iterations the VAE is finetuned on evaluated tracks, all elite descriptors are recomputed, the archive is remapped (novelty‑filtered) and the novelty threshold is auto‑tuned toward a target archive size (CSC controller from Grillotti & Cully, 2022)
- Analyze: stats, elite track images, t‑SNE grids; export to visualizer


## Project layout

- `qd/` — the QD pipeline: `ns.py` / `novelty_search.ipynb` (entry point), `qd_runner.py` (main loop), `emitter.py`, `evaluator.py`, `config.py`, `qd_stats.py`, `archive_visualizer.py`
- `qd/vae/` — telemetry VAE: model, data, losses, preprocessing, training (used both pretrained and finetuned online)
- `qd/pretrained_models/` — checkpointed VAEs (telemetry-metrics and XML variants) plus their evaluation notebooks
- `qd/datasets/` — dataset generators for VAE training data and precomputed embeddings
- `qd/analysis/` — results analysis notebooks/scripts and concept-figure generators
- `src/` — Node simulation API (`sim/mapElitesAPI.js`), JS genotype/crossover/mutation code, the TORCS source + Dockerfile, and Python telemetry tools (`sirianni_tools/`)
- `web/` — the web visualizer (Express server + static front end, deployed to Netlify)


## How to run
Requires Docker, Python >= 3.12.10 and node >= v24.10.0

1. Install Python dependencies (from the root folder):
    - `pip install -e .[dev]`
2. Build the simulation backend (from the `src` folder):
    - `docker build -t torcs:dev .` (build the TORCS image)
    - `npm install` (install the simulation API dependencies)
3. Start the track generation/simulation API (it must be running before the algorithm starts):
    - from the root folder: `node src/sim/mapElitesAPI.js` (listens on port 4242)
4. Run the algorithm (from the root folder, so `data/` and `qd/` paths resolve):
    - run the script `python qd/ns.py`

Runs checkpoint themselves periodically (archive, stats, buffer, finetuned VAE) under `data/ns/`; restarting resumes from the latest checkpoint automatically. Tunables (iterations, batch size, retraining cadence, novelty threshold, target archive size, latent/measure dim) live in `qd/config.py`.

### Visualize results

- **Web visualizer** (real-time browse/debug, same UI as the [live demo](https://pcgtrack.netlify.app/)): from the `web` folder run `npm install` then `npm start` and open `http://localhost:3000`.
- **Archive plots** (stats, elite track images, finetuning curves): produced automatically during a run. Additional analyses (UMAP heatmaps, t-SNE grids) are available on demand from the analysis notebooks in `qd/analysis/`.

## License

MIT License.
