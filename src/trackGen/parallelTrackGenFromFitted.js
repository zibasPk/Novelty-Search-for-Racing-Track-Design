import fs from 'fs';
import path from 'path';
import { generateTrack } from './trackGenerator.js';
import log from "loglevel";
import { BBOX } from '../utils/constants.js';

const FITTED_DIR = './data/voronoi/fitted';
const CONCURRENCY_LIMIT = 20;
const OUTPUT_FILE = './tracks_mixedRng.json';

log.setLevel('info');

async function runGenFromFitted(filePath) {
    const raw = fs.readFileSync(filePath, 'utf-8');
    const data = JSON.parse(raw);

    const { id, mode, trackSize, rngMode, dataSet, selectedCells } = data;

    const { track } = await generateTrack({
        mode,
        bbox: BBOX,
        seed: id,
        trackSize,
        saveJSON: false,
        dataSet,
        selected: selectedCells,
        rngMode
    });

    // Convert track from [{x, y}] to [[x, y]]
    const convertedTrack = track.map(point => [point.x, point.y]);

    return { seed: id, track: convertedTrack };
}

async function runAllGens() {
    // List all JSON files in the fitted directory
    const files = fs.readdirSync(FITTED_DIR)
        .filter(f => f.endsWith('.json'))
        .map(f => path.join(FITTED_DIR, f));

    log.info(`Found ${files.length} fitted files in ${FITTED_DIR}`);

    const allTracks = [];
    const genPromises = [];

    for (let i = 0; i < files.length; i++) {
        const filePath = files[i];
        const promise = runGenFromFitted(filePath)
            .then(result => {
                allTracks.push(result);
            })
            .catch(err => {
                log.error(`Error processing ${filePath}: ${err.message}`);
            });

        genPromises.push(promise);

        if (genPromises.length >= CONCURRENCY_LIMIT) {
            await Promise.all(genPromises);
            genPromises.length = 0;
            log.info(`Progress: ${allTracks.length} / ${files.length} tracks processed`);
        }
    }

    await Promise.all(genPromises); // Await any remaining

    // Sort by seed
    allTracks.sort((a, b) => a.seed - b.seed);

    log.info(`Writing ${allTracks.length} tracks to ${OUTPUT_FILE}`);

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

    log.info(`All gens completed. Saved ${allTracks.length} tracks to ${OUTPUT_FILE}`);
}

runAllGens().catch(err => log.error(`Unexpected error: ${err.message}`));
