//this is a variant of parallel test to tests tracks with seeds from 0 to 99 
//with fixed params

import { simulate } from './simulateTrack.js';
import { JSON_DEBUG, SIMULATION_TIMEOUT, } from '../utils/constants.js';
import { SimulationTimeoutError } from '../utils/errors.js';

const STARTING_SEED = 0;
const TOTAL_UNIQUE_TRACKS = 2000;
const REPETITIONS_PER_TRACK = 1;
const CONCURRENCY_LIMIT = 20; // Number of parallel simulations


let timedOutSeeds = new Set();

async function runSimulation(simulationIndex) {
  try {
    console.log(`Starting simulation ${simulationIndex}`);
    let startTime = Date.now();

    // Generate random parameters for the simulation
    const mode = 'voronoi';
    const seed = simulationIndex;
    const trackSize = (simulationIndex % 8) + 1;

    // Run the simulation
    const { fitness } = await simulate(mode, trackSize, [], [], seed, JSON_DEBUG, false);

    console.log(`Simulation ${simulationIndex} completed. Fitness:`, fitness);
    let endTime = Date.now();
    console.log(`Simulation ${simulationIndex} execution time: ${(endTime - startTime) / 1000} seconds`);
    return fitness;
  } catch (error) {
    console.error(`Error in simulation ${simulationIndex}: ${error.message}`);
    if (error instanceof SimulationTimeoutError) {
      timedOutSeeds.add(simulationIndex);
    }
  } 
}

async function runSimulations() {
  const startTime = Date.now();
  const simulationPromises = [];
  for (let i = STARTING_SEED; i < REPETITIONS_PER_TRACK * TOTAL_UNIQUE_TRACKS + STARTING_SEED; i++) {
    simulationPromises.push(runSimulation(i % TOTAL_UNIQUE_TRACKS));
    if (simulationPromises.length >= CONCURRENCY_LIMIT) {
      await Promise.all(simulationPromises);
      simulationPromises.length = 0;
    }
  }
  await Promise.all(simulationPromises); // Await any remaining simulations
  const endTime = Date.now();
  console.log(`All simulations completed in ${(endTime - startTime) / 1000} seconds`);
  if (timedOutSeeds.size > 0) {
    console.log(`Simulations timed out for seeds: ${Array.from(timedOutSeeds).join(', ')}`);
  }
}


runSimulations().catch(err => console.error(`Unexpected error: ${err.message}`));