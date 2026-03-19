import express from 'express';
import cors from 'cors';
import { generateTrack } from '../trackGen/trackGenerator.js';
import { crossover, crossover2 } from '../genetic/crossoverVoronoi.js';
import { crossover as crossoverConvexHull } from '../genetic/crossoverConvexHull.js';
import { mutationConvexHull, mutationVoronoi } from '../genetic/mutation.js';
import { BBOX, JSON_DEBUG, LOG_DIR } from '../utils/constants.js';
import { simulate } from './simulateTrack.js';
import { initLogger } from '../utils/logger.js';


// setup logging
let dateTime = new Date().toISOString().replace(/:/g, '-');
let logPath = LOG_DIR + `Api_${dateTime}.log`;

let log = initLogger({
  filePath: logPath,
  level: "debug",
  withTimestamp: true
});


const app = express();

// --- 2. ENABLE CORS HERE ---
// This allows frontends to access this API ( both web and jupyter notebook )
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:8888']
}));
// ---------------------------

app.use(express.json());

/* ─────────────────────────────────────────────────────────────
   Helpers
   ──────────────────────────────────────────────────────────── */
const safeArray = arr => (Array.isArray(arr) ? arr : []);

/* ─────────────────────────────────────────────────────────────
   /generate
   ──────────────────────────────────────────────────────────── */
app.post('/generate', async (req, res) => {
  try {
    const { id, mode, trackSize, rngMode } = req.body;

    const { track, generator, splineVector } =
      await generateTrack({ mode, bbox: BBOX, seed: id, trackSize, saveJSON: JSON_DEBUG, rngMode });

    const response = {
      id,
      mode,
      dataSet: generator.dataSet,
      selectedCells: safeArray(generator.selectedCells).map(cell => ({
        x: cell.site.x,
        y: cell.site.y
      })),
      trackSize: generator.trackSize,
      splineVector
    };

    res.json(response);
  } catch (error) {
    log.error(`/generate for ${req.body.id}:`, error.message);
    res.status(500).json({ error: error.message });
  }
});


app.post('/genforweb', async (req, res) => {
  try {
    const { id, mode, trackSize , perlin_parameters, rngMode} = req.body;

    const { track, generator, splineVector } = await generateTrack({
      mode,
      bbox: BBOX,
      seed: id,
      trackSize,
      saveJSON: false,
      rngMode: rngMode,
      perlin_parameters
    });

    const response = {
      mode,
      track,
      generator: generator.toJSON(),
      splineVector
    };

    res.json(response);
  } catch (error) {
    log.error(`/genforweb for ${req.body.id}:`, error.message);
    res.status(500).json({ error: error.message });
  }
});

/* ─────────────────────────────────────────────────────────────
   /reconstruct
   Rebuild a track from its genotype (mode, dataSet, selectedCells, trackSize)
   and return the spline points + generator metadata.
   ──────────────────────────────────────────────────────────── */
app.post('/reconstruct', async (req, res) => {
  try {
    const { mode, seed, dataSet, selectedCells, trackSize } = req.body;

    if (!mode || !dataSet) {
      return res.status(400).json({ error: 'mode and dataSet are required' });
    }

    const sel = safeArray(selectedCells);

    const timeout = 5000;
    const { track, generator, splineVector } = await Promise.race([
      generateTrack({
        mode,
        bbox: BBOX,
        seed,
        trackSize,
        saveJSON: false,
        dataSet,
        selected: sel
      }),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Track generation timed out')), timeout))
    ]);

    res.json({
      mode,
      track,                           // full spline-smoothed point list
      splineVector,                    // resampled fixed-length vector
      trackSize: generator.trackSize,
      dataSet: generator.dataSet,
      edges: generator.diagram.edges,
      selectedCells: safeArray(generator.selectedCells).map(cell => ({
        x: cell.site.x,
        y: cell.site.y
      }))
    });
  } catch (error) {
    log.error('/reconstruct error:', error.message);
    res.status(500).json({ error: error.message });
  }
});

/* ─────────────────────────────────────────────────────────────
   /evaluate
   ──────────────────────────────────────────────────────────── */
app.post('/evaluate', async (req, res) => {
  try {
    const { id, mode, dataSet, selectedCells, rngMode } = req.body;

    const sel = safeArray(selectedCells);
    const simulationResult = await simulate(
      mode,
      sel.length,
      dataSet,
      sel,
      id,
      JSON_DEBUG,
      false,
      rngMode
    );

    res.json({
      fitness: simulationResult.fitness,
      splineVector: simulationResult.splineVector
    });
    let resultsToPrint = simulationResult.fitness;
    if (resultsToPrint.embedding_data) {
      resultsToPrint.embedding_data = simulationResult.fitness.embedding_data.length;
    }
    log.info('Returning fitness from /evaluate: ',
      JSON.stringify(resultsToPrint));
  } catch (error) {
    log.error(`/evaluate for ${req.body.id}:`, error.message);
    res.status(500).json({ error: error.message });
  }
});

/* ─────────────────────────────────────────────────────────────
   /crossover
   ──────────────────────────────────────────────────────────── */
app.post('/crossover', async (req, res, next) => {
  log.info('Crossover endpoint called');
  try {
    const { parent1, parent2, mode, genetic_seed } = req.body;
    if (!parent1 || !parent2 ||
      !parent1.dataSet || !parent2.dataSet) {
      return res.status(400).json({ error: 'Invalid parent data' });
    }

    const timeout = 5000;

    const [result1, result2] = await Promise.all([
      Promise.race([
        generateTrack({
          mode,
          bbox: BBOX,
          seed: parent1.id,
          trackSize: parent1.trackSize,
          saveJSON: false,
          dataSet: parent1.dataSet,
          selected: safeArray(parent1.selectedCells)
        }),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Track generation timed out')), timeout))
      ]),
      Promise.race([
        generateTrack({
          mode,
          bbox: BBOX,
          seed: parent2.id,
          trackSize: parent2.trackSize,
          saveJSON: false,
          dataSet: parent2.dataSet,
          selected: safeArray(parent2.selectedCells)
        }),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Track generation timed out')), timeout))
      ])
    ]);

    const trackGenerator1 = result1.generator;
    const trackGenerator2 = result2.generator;

    if (mode === 'voronoi') {
      log.debug('CROSSOVER VORONOI');
      try {
        const result = Math.random() < 0 //mix between two crossovers , 0.5 to balance, 1 for only crossover method 1 , 0 only method 2 
          ? crossover(trackGenerator1, trackGenerator2, true, genetic_seed)
          : crossover2(trackGenerator2, trackGenerator1, true, genetic_seed);

        log.debug("Dataset lenght: ", result.ds.length);
        log.debug("Selected cells lenght: ", result.sel.length);

        return res.json({ offspring: { ds: result.ds, sel: result.sel } });
      } catch (err) {
        log.error('Error during crossover:', err.message);
        log.error('Parent 1:', parent1.id);
        log.error('Parent 2:', parent2.id);
        return res.status(500).json({ error: 'Crossover failed.' });
      }
    }

    /* convexHull crossover */
    const result = crossoverConvexHull(trackGenerator1, trackGenerator2, true);
    res.json({ offspring: { ds: result.ds } });
  } catch (error) {
    log.error(`/crossover for ${req.body.parent1.id} and ${req.body.parent2.id}:`, error);
    next(error);
  }
});

/* ─────────────────────────────────────────────────────────────
   /mutate
   ──────────────────────────────────────────────────────────── */
app.post('/mutate', async (req, res, next) => {
  try {
    const { individual, intensityMutation = 50, genetic_seed } = req.body;
    if (!individual || !individual.dataSet) {
      return res.status(400).json({ error: 'Invalid individual data' });
    }

    const timeout = 5000;
    const { generator: trackGenerator } = await Promise.race([
      generateTrack({
        mode: individual.mode,
        bbox: BBOX,
        seed: individual.id,
        trackSize: individual.trackSize,
        saveJSON: false,
        dataSet: individual.dataSet,
        selected: safeArray(individual.selectedCells)
      }),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Track generation timed out')), timeout))
    ]);


    let mutationSeed = genetic_seed; // we use the id because its generated at random beforehand. Watchout same ids will generate the same mutation.

    if (individual.mode === 'voronoi') {
      const mutatedData = mutationVoronoi(trackGenerator, intensityMutation, mutationSeed);
      return res.json({
        mutated: { dataSet: mutatedData.ds, selectedCells: mutatedData.sel }
      });
    }

    if (individual.mode === 'convexHull') {
      const mutatedData = mutationConvexHull(trackGenerator, intensityMutation, mutationSeed);
      return res.json({ mutated: { dataSet: mutatedData.ds } });
    }

    res.status(400).json({ error: 'Invalid track generation mode in /mutate' });
  } catch (error) {
    log.error(`/mutate for ${req.body.individual.id}:`, error.message);
    next(error);
  }
});



/* ─────────────────────────────────────────────────────────────
   Global error handler
   ──────────────────────────────────────────────────────────── */
app.use((error, req, res, next) => {
  log.error('Error:', error);
  res.status(500).json({ error: 'Internal server error' });
});

/* ─────────────────────────────────────────────────────────────
   Start server
   ──────────────────────────────────────────────────────────── */
const PORT = 4242;
app.listen(PORT, () => {
  log.info(`MAP-Elites API running on port ${PORT}`);
});