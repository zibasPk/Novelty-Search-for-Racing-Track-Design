import { simulate } from './simulateTrack.js';
import { JSON_DEBUG, SIMULATION_TIMEOUT, LOG_DIR } from '../utils/constants.js';
import { initLogger } from '../utils/logger.js';

const TOTAL_SIMULATIONS = 20000;
const CONCURRENCY_LIMIT = 20; // Number of parallel simulations

async function runSimulation(simulationIndex) {
  try {
    log.info(`Starting simulation ${simulationIndex}`);

    // Generate random parameters for the simulation
    const mode = Math.random() < 0.5 ? 'voronoi' : 'convexHull';
    const trackSize = mode === 'voronoi' ? Math.ceil(Math.random() * 4) + 1 : 50;
    const seed = Math.random();

    // Run the simulation
    const { fitness } = await Promise.race([
      simulate(mode, trackSize, [], [], seed, JSON_DEBUG, false),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Simulation timeout')), SIMULATION_TIMEOUT)
      )
    ]);

    log.info(`Simulation ${simulationIndex} completed.`);
    log.trace(`fitness: ${JSON.stringify(fitness)}`);
    return fitness;
  } catch (error) {
    log.error(`Error in simulation ${simulationIndex}: ${error.message}`);
  }
}

async function runSimulations() {
  const simulationPromises = [];
  for (let i = 0; i < TOTAL_SIMULATIONS; i++) {
    simulationPromises.push(runSimulation(i + 1));
    if (simulationPromises.length >= CONCURRENCY_LIMIT) {
      await Promise.all(simulationPromises);
      simulationPromises.length = 0;
    }
  }
  await Promise.all(simulationPromises); // Await any remaining simulations
}


// setup logging
let dateTime = new Date().toISOString().replace(/:/g, '-');
let logPath = LOG_DIR +`ParallelTest_${dateTime}.log`;

let log = initLogger({
  filePath: logPath,
  level: "info",
  withTimestamp: true
});

runSimulations().catch(err => log.error(`Unexpected error: ${err.message}`));