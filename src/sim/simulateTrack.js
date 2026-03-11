import { exec } from 'child_process';
import { promises as fs } from 'fs';
import { generateTrack } from '../trackGen/trackGenerator.js';
import { TorcsXMLGenerator } from '../trackGen/torcsXMLGenerator.js';
import { saveFitnessToJson } from '../utils/jsonUtils.js';
import path from 'path';
import os from 'os';
import {
  BBOX,
  MODE,
  DOCKER_IMAGE_NAME,
  MEMORY_LIMIT,
  SIMULATION_TIMEOUT,
  TARGET_RACE_DURATION,
  DEFAULT_REPETITIONS,
  OUTPUT_DIR_XML
} from '../utils/constants.js';
import { SimulationTimeoutError } from '../utils/errors.js';
import log from "loglevel";

const executeCommand = (command) => {
  return new Promise((resolve, reject) => {
    exec(command, { maxBuffer: 1024 * 1024 * 10 }, (error, stdout, stderr) => {
      if (error) {
        reject(new Error(`${error.message}`));
        return;
      }
      if (stderr) {
        log.warn(`stderr: ${stderr}`);
      }
      resolve(stdout.trim());
    });
  });
};

export async function simulate(
  mode = MODE,
  trackSize = 0,
  dataSet = [],
  selected = [],
  seed = null,
  saveJson = false,
  plot = false,
  rngMode,
) {
  if (isNaN(trackSize)) {
    if (mode === 'voronoi') {
      if (selected.length > 0) {
        trackSize = selected.length;
      }
    } else {
      trackSize = 50;
    }
  }

  // generate track json
  const trackResults = await generateTrack({
    mode, bbox: BBOX, seed, trackSize,
    saveJSON: saveJson, dataSet, selected, rngMode
  });
  

  // translate to XML for TORCS
  const xmlGenerator = new TorcsXMLGenerator(trackResults.track, seed);
  const trackXml = xmlGenerator.generateXML(0, true);
  log.info(`SEED: ${seed}`);
  log.info(`MODE: ${mode}`);
  log.info(`trackSize: ${trackSize}`);
  log.info('TrackLength:', xmlGenerator.getLength().toFixed(2));

  let containerId;
  let timeoutId;

  try {
    log.debug(`Starting Docker container for simulation ${seed}`);
    containerId = await startDockerContainer(seed);

    // Move track XML to Docker container and generate track files
    log.debug(`Generating and moving track files to Docker container for ${seed}`);
    const trackGenOutput = await generateAndMoveTrackFiles(containerId, trackXml, seed);

    const simCommand = `docker exec ${containerId} python3 /usr/local/lib/sirianni_tools/run_simulations.py --track-export --repetitions ${DEFAULT_REPETITIONS} -d ${TARGET_RACE_DURATION} --json ${plot ? '--plots' : ''} -e`;
    const simulationOutput = await Promise.race([
      executeCommand(simCommand),
      new Promise((_, reject) =>
        timeoutId = setTimeout(() => reject(new SimulationTimeoutError()), SIMULATION_TIMEOUT)
      )
    ]);

    let rawMetrics = {};
    const jsonStart = simulationOutput.indexOf('===FINAL_JSON_START===');
    const jsonEnd = simulationOutput.indexOf('===FINAL_JSON_END===', jsonStart);
    if (jsonStart !== -1 && jsonEnd !== -1) {
      const jsonString = simulationOutput
        .substring(jsonStart + '===FINAL_JSON_START==='.length, jsonEnd)
        .trim();
      rawMetrics = JSON.parse(jsonString);
    } else {
      throw new Error('JSON markers not found in run_simulations.py output.');
    }
    const { length, deltaX, deltaY, deltaAngleDegrees } = parseTrackgenOutput(trackGenOutput);
    const {
      speed_entropy = 0,
      acceleration_entropy = 0,
      braking_entropy = 0,
      positions_mean = 0,
      avg_radius_mean = 0,
      gaps_mean = 0,
      right_bends = 0,
      avg_radius_var = 0,
      total_overtakes = 0,
      straight_sections = 0,
      gaps_var = 0,
      left_bends = 0,
      positions_var = 0,
      curvature_entropy = 0,
      ...rest
    } = rawMetrics;
    const fitness = {
      length,
      deltaX,
      deltaY,
      deltaAngleDegrees,
      speed_entropy,
      acceleration_entropy,
      braking_entropy,
      positions_mean,
      avg_radius_mean,
      gaps_mean,
      right_bends,
      avg_radius_var,
      total_overtakes,
      straight_sections,
      gaps_var,
      left_bends,
      positions_var,
      curvature_entropy,
      ...rest
    };
    if (saveJson) {
      await saveFitnessToJson(
        seed,
        mode,
        trackResults.generator.trackSize,
        rngMode,
        fitness
      );
    }
    return { fitness: fitness, splineVector: trackResults.splineVector };
  } catch (err) {
    throw err;
  } finally {
    clearTimeout(timeoutId);
    if (containerId) {
      await stopDockerContainer(containerId);
    }
  }
}

async function startDockerContainer(seed) {
  let randomSuffix = Math.random().toString(36).substring(2, 7);
  let containterName = "track_simulation_" + seed + "_" + randomSuffix;
  const containerId = await executeCommand(
    `docker run -d --memory ${MEMORY_LIMIT} --name ${containterName} ${DOCKER_IMAGE_NAME} sleep infinity`
  );
  log.info(`Docker container started with ID: ${containerId}`);
  await executeCommand(
    `docker exec ${containerId} mkdir -p /usr/local/share/games/torcs/tracks/road/output`
  );
  return containerId;
}

async function stopDockerContainer(containerId) {
  try {
    await executeCommand(`docker rm --force ${containerId}`);
    log.info(`Docker container ${containerId} stopped and removed.`);
  } catch (err) {
    log.error(`Failed to stop Docker container ${containerId}: ${err.message}`);
  }
}

async function generateAndMoveTrackFiles(containerId, trackXml, seed) {
  const tmpDir = os.tmpdir();
  const tmpFilePath = path.join(tmpDir, `${seed}.xml`);
  await fs.writeFile(tmpFilePath, trackXml);

  try {
    await executeCommand(
      `docker cp ${tmpFilePath} ` +
      `${containerId}:/usr/local/share/games/torcs/tracks/road/output/${seed}.xml`
    );
    await executeCommand(
      `docker exec ${containerId} ` +
      `mv /usr/local/share/games/torcs/tracks/road/output/${seed}.xml ` +
      `/usr/local/share/games/torcs/tracks/road/output/output.xml`
    );
    const trackgenOutput = await executeCommand(
      `docker exec ${containerId} xvfb-run trackgen -c road -n output`
    );
    await fs.unlink(tmpFilePath);
    return trackgenOutput;
  } catch (err) {
    await fs.unlink(tmpFilePath);
    throw new Error(`Failed to generate and move track files: ${err.message}`);
  }
}

function parseTrackgenOutput(trackgenOutput) {
  const lengthMatch = trackgenOutput.match(/length\s*=\s*([\d.]+)/);
  const deltaXMatch = trackgenOutput.match(/Delta X\s*=\s*(-?[\d.]+)/);
  const deltaYMatch = trackgenOutput.match(/Delta Y\s*=\s*(-?[\d.]+)/);
  const deltaAngleMatch = trackgenOutput.match(/Delta Ang\s*=\s*(-?[\d.]+)\s*\((-?[\d.]+)\)/);
  return {
    length: lengthMatch ? parseFloat(lengthMatch[1]) : null,
    deltaX: deltaXMatch ? parseFloat(deltaXMatch[1]) : null,
    deltaY: deltaYMatch ? parseFloat(deltaYMatch[1]) : null,
    deltaAngleDegrees: deltaAngleMatch ? parseFloat(deltaAngleMatch[2]) : null
  };
}

if (process.argv[1].includes('simulateTrack.js')) {
  simulate().catch(err => log.error(`Unhandled error: ${err.message}`));
}
