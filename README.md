# Quality Diversity for Racing Track Design

MSc thesis project to evolve diverse, high‑quality racing tracks with Quality‑Diversity search, run headless sims, and analyze with a data‑driven stack. Take inspiration from the approach; it’s a thesis prototype, not a product. The full methodology, engineering trade‑offs, and TORCS quirks are in the thesis.

- [Blog article](https://blog.martino.im/Quality-Diversity-for-Racing-Tracks-Design)
- [Executive summary](https://blog.martino.im/qd_executive_summary.pdf)
- [Full thesis](https://blog.martino.im/quality_diversity.pdf)
- [Live visualizer](https://pcgtrack.netlify.app/)

## What it is

- Quality‑Diversity search (Novelty Search with local competition over an unstructured archive) in a learned behavior space — no hand‑tuned features
- Behavior descriptors from a **VAE trained on driving telemetry**, finetuned online during the run (AURORA‑style) so the latent space adapts to the tracks being discovered
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
- Analyze: stats, heatmaps, elite track images, t‑SNE grids; export to visualizer


## How to run
Requires Docker, Python >= 3.12.10 and node >= v24.10.0

1. In root folder:
    - `pip install -e .[dev]` (to install python dependencies)
2. In `src` folder:
    - `docker build -t torcs:dev .` (to build TORCS image)
    - `npm install` (install dependencies for api)
    - `node sim/mapElitesAPI.js` (start the track generation/simulation API on port 4242)
3. To run the algorithm use the notebook `qd/novelty_search.ipynb` or run the script `qd/ns.py`.

Runs checkpoint themselves periodically (archive, stats, finetuned VAE) under `data/ns/`; restarting the script resumes from the latest checkpoint automatically. Tunables (iterations, batch size, retraining cadence, novelty threshold, target archive size) live in `qd/config.py`.


## Borrowable bits

- Voronoi genotype + crossovers (Random‑Line Partitioning, Relative Reconstruction)
- Normalized overtakes (robust to geometry artifacts)
- Telemetry‑VAE behavior space with online finetuning + archive remap (no “#turns” features)
- Self‑tuning novelty threshold to hold a target archive size
- Per‑sim container pattern for isolation and easy parallelism

## Caveats

- TORCS is dated (no elevation/banking).
- Per‑sim containers add overhead by design.
- Metrics need normalization and validity checks.

## Roadmap (ideas to steal)

- Modern engine (Unity/Unreal) for elevation, banking, richer geometry
- Surrogate models to reduce sim cost
- Designer‑in‑the‑loop for “fun”/aesthetics
- Advanced QD (e.g., DCG‑MAP‑Elites), hybrid gradient methods
- Stronger geometry validity (self‑intersection, curvature/grade limits)

## License

MIT License.
