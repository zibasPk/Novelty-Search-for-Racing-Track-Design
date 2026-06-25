//this is a variant of parallel test to tests tracks with seeds from 0 to 99 
//with fixed params

import { simulate } from './simulateTrack.js';
import { JSON_DEBUG, SIMULATION_TIMEOUT, LOG_DIR, RngMode } from '../utils/constants.js';
import { SimulationTimeoutError } from '../utils/errors.js';
import { initLogger } from '../utils/logger.js';

const STARTING_SEED = 0;
const ENDING_SEED = 20000;
const CONCURRENCY_LIMIT = 20; // Number of parallel simulations


let timedOutSeeds = new Set();

async function runSimulation(simulationIndex) {
  try {
    log.info(`Starting simulation ${simulationIndex}`);
    let startTime = Date.now();

    // Generate random parameters for the simulation
    const mode = 'voronoi';
    const seed = simulationIndex;
    // tracksize between 4 and 10
    const trackSize = simulationIndex % 7 + 4; // This will cycle through values from 4 to 10
    let rngMode = seed % 2 === 0 ? RngMode.UNIFORM : RngMode.PERLIN; // Alternate between 'uniform' and 'perlin' RNG modes
    // Run the simulation
    const { fitness } = await simulate({
      mode,
      trackSize,
      selected: [],
      dataSet: [],
      seed,
      saveJson: JSON_DEBUG,
      plot: false,
      rngMode
    });

    log.trace(`fitness: ${JSON.stringify(fitness)}`);
    let endTime = Date.now();
    log.info(`Simulation ${simulationIndex} completed in: ${(endTime - startTime) / 1000} seconds`);
    return fitness;
  } catch (error) {
    log.error(`Error in simulation ${simulationIndex}: ${error.message}`);
    if (error instanceof SimulationTimeoutError) {
      timedOutSeeds.add(simulationIndex);
    }
  } 
}

async function runSimulations() {
  const startTime = Date.now();
  const simulationPromises = [];
  for (let i = STARTING_SEED; i < ENDING_SEED; i++) {
    simulationPromises.push(runSimulation(i));
    if (simulationPromises.length >= CONCURRENCY_LIMIT) {
      await Promise.all(simulationPromises);
      simulationPromises.length = 0;
    }
  }
  await Promise.all(simulationPromises); // Await any remaining simulations
  const endTime = Date.now();
  log.info(`All simulations completed in ${(endTime - startTime) / 1000} seconds`);
  if (timedOutSeeds.size > 0) {
    log.info(`Simulations timed out for seeds: ${Array.from(timedOutSeeds).join(', ')}`);
  }
}

let dateTime = new Date().toISOString().replace(/:/g, '-');
let logPath = LOG_DIR +`ParallelTest_${dateTime}.log`;

let log = initLogger({
  filePath: logPath,
  level: "info",
  withTimestamp: true
});
runSimulations().catch(err => console.error(`Unexpected error: ${err.message}`));