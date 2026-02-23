import fs from 'fs';
import { generateTrack } from './trackGenerator.js';
import log from "loglevel";
import {
  BBOX
} from '../utils/constants.js';

const STARTING_SEED = 0;
const TOTAL_UNIQUE_TRACKS = 20000;
const REPETITIONS_PER_TRACK = 1;
const CONCURRENCY_LIMIT = 20; // Number of parallel simulations
const OUTPUT_FILE = './tracks.json'; // Single output file

async function runGen(genIndex) {
    const mode = 'voronoi';
    const seed = genIndex;
    const trackSize = (genIndex % 8) + 1;
    
    // Run the simulation
    const { track, generator, splineVector } =
          await generateTrack({ mode, bbox: BBOX, seed, trackSize, saveJSON: false });
    
    // Convert track from [{x, y}] to [[x, y]]
    const convertedTrack = track.map(point => [point.x, point.y]);
    
    return { seed, track: convertedTrack };
}

async function runGens() {
    const allTracks = [];
    const genPromises = [];
    
    for (let i = STARTING_SEED; i < REPETITIONS_PER_TRACK * TOTAL_UNIQUE_TRACKS + STARTING_SEED; i++) {
        const promise = runGen(i % TOTAL_UNIQUE_TRACKS)
            .then(result => {
                allTracks.push(result);
            })
            .catch(err => log.error`Error processing seed ${i}: ${err.message}`);
        
        genPromises.push(promise);
        
        if (genPromises.length >= CONCURRENCY_LIMIT) {
            await Promise.all(genPromises);
            genPromises.length = 0;
        }
    }
    
    await Promise.all(genPromises); // Await any remaining simulations
    
    // Sort by seed
    allTracks.sort((a, b) => a.seed - b.seed);
    
    log.info`Writing ${allTracks.length} tracks to ${OUTPUT_FILE}`;
    
    // Write using streaming to avoid string length limits
    const writeStream = fs.createWriteStream(OUTPUT_FILE);
    
    writeStream.write('[');
    
    for (let i = 0; i < allTracks.length; i++) {
        const trackJson = JSON.stringify(allTracks[i]);
        writeStream.write(trackJson);
        
        if (i < allTracks.length - 1) {
            writeStream.write(',');
        }
    }
    
    writeStream.write(']');
    writeStream.end();
    
    // Wait for the stream to finish
    await new Promise((resolve, reject) => {
        writeStream.on('finish', resolve);
        writeStream.on('error', reject);
    });
    
    log.info`All gens completed. Saved ${allTracks.length} tracks to ${OUTPUT_FILE}`;
}

runGens().catch(err => log.error`Unexpected error: ${err.message}`);