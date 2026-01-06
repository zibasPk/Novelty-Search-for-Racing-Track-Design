import express from 'express';
import cors from 'cors'; // <--- 1. IMPORT CORS HERE
import { generateTrack } from '../trackGen/trackGenerator.js';
import { crossover, crossover2 } from '../genetic/crossoverVoronoi.js';
import { crossover as crossoverConvexHull } from '../genetic/crossoverConvexHull.js';
import { mutationConvexHull, mutationVoronoi } from '../genetic/mutation.js';
import { BBOX, JSON_DEBUG,LOG_DIR } from '../utils/constants.js';
import { simulate } from './simulateTrack.js';
import { initLogger } from '../utils/logger.js';


// setup logging
let dateTime = new Date().toISOString().replace(/:/g, '-');
let logPath = LOG_DIR +`Api_${dateTime}.log`;

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
    const { id, mode, trackSize } = req.body;

    const { track, generator, splineVector } =
      await generateTrack(mode, BBOX, id, trackSize, JSON_DEBUG);

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
    log.error(`/generate for ${req.body.id}:`, error);
    res.status(500).json({ error: error.message });
  }
});


app.post('/genforweb', async (req, res) => {
  try {
    const { id, mode, trackSize } = req.body;

    const { track, generator, splineVector } =
      await generateTrack(mode, BBOX, id, trackSize, JSON_DEBUG);

    const response = {
      mode,
      track,
      generator: generator.toJSON(),
      splineVector
    };

    res.json(response);
  } catch (error) {
    log.error(`/generate for ${req.body.id}:`, error);
    res.status(500).json({ error: error.message });
  }
});

/* ─────────────────────────────────────────────────────────────
   /evaluate
   ──────────────────────────────────────────────────────────── */
app.post('/evaluate', async (req, res) => {
  try {
    const { id, mode, dataSet, selectedCells } = req.body;

    const sel = safeArray(selectedCells);
    const simulationResult = await simulate(
      mode,
      sel.length,
      dataSet,
      sel,
      id,
      JSON_DEBUG
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
    log.error(`/evaluate for ${req.body.id}:`, error);
    res.status(500).json({ error: error.message });
  }
});

/* ─────────────────────────────────────────────────────────────
   /crossover
   ──────────────────────────────────────────────────────────── */
app.post('/crossover', async (req, res, next) => {
  log.info('Crossover endpoint called');
  try {
    const { parent1, parent2, mode } = req.body;
    if (!parent1 || !parent2 ||
      !parent1.dataSet || !parent2.dataSet) {
      return res.status(400).json({ error: 'Invalid parent data' });
    }

    const timeout = 5000;

    const [result1, result2] = await Promise.all([
      Promise.race([
        generateTrack(
          mode,
          BBOX,
          parent1.id,
          parent1.trackSize,
          false,
          parent1.dataSet,
          safeArray(parent1.selectedCells)
        ),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('Track generation timed out')), timeout))
      ]),
      Promise.race([
        generateTrack(
          mode,
          BBOX,
          parent2.id,
          parent2.trackSize,
          false,
          parent2.dataSet,
          safeArray(parent2.selectedCells)
        ),
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
          ? crossover(trackGenerator1, trackGenerator2, true)
          : crossover2(trackGenerator2, trackGenerator1, true);

        log.debug("Dataset lenght: ", result.ds.length);
        log.debug("Selected cells lenght: ", result.sel.length);

        return res.json({ offspring: { ds: result.ds, sel: result.sel } });
      } catch (err) {
        log.error('Error during crossover:', err);
        log.error('Parent 1:', JSON.stringify(parent1, null, 2));
        log.error('Parent 2:', JSON.stringify(parent2, null, 2));
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
    const { individual, intensityMutation = 50 } = req.body;
    if (!individual || !individual.dataSet) {
      return res.status(400).json({ error: 'Invalid individual data' });
    }

    const timeout = 5000;
    const { generator: trackGenerator } = await Promise.race([
      generateTrack(
        individual.mode,
        BBOX,
        individual.id,
        individual.trackSize,
        false,
        individual.dataSet,
        safeArray(individual.selectedCells)
      ),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Track generation timed out')), timeout))
    ]);

    if (individual.mode === 'voronoi') {
      const mutatedData = mutationVoronoi(trackGenerator, intensityMutation);
      return res.json({
        mutated: { dataSet: mutatedData.ds, selectedCells: mutatedData.sel }
      });
    }

    if (individual.mode === 'convexHull') {
      const mutatedData = mutationConvexHull(trackGenerator, intensityMutation);
      return res.json({ mutated: { dataSet: mutatedData.ds } });
    }

    res.status(400).json({ error: 'Invalid track generation mode in /mutate' });
  } catch (error) {
    log.error(`/mutate for ${req.body.individual.id}:`, error);
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