
import { simulate } from '../../sim/simulateTrack.js';
import { JSON_DEBUG, SIMULATION_TIMEOUT, } from '../constants.js';
import { SimulationTimeoutError } from '../errors.js';
import log from "loglevel";


async function runSimulation(simulationIndex) {
  try {
    log.info(`Starting simulation ${simulationIndex}`);
    let startTime = Date.now();

    // Generate random parameters for the simulation
    const mode = 'voronoi';
    const seed = simulationIndex;
    const trackSize = (simulationIndex % 8) + 1;

    // Run the simulation
    const { fitness } = await simulate(mode, trackSize, [], [], seed, JSON_DEBUG, false);

    log.info(`Simulation ${simulationIndex} completed. Fitness:`, fitness);
    let endTime = Date.now();
    log.info(`Simulation ${simulationIndex} execution time: ${(endTime - startTime) / 1000} seconds`);
    return fitness;
  } catch (error) {
    log.error(`Error in simulation ${simulationIndex}: ${error.message}`);
  } 
}

log.setLevel("debug");
let fitness = await runSimulation(100);

log.info(`Final fitness for simulation 195: ${fitness}`);