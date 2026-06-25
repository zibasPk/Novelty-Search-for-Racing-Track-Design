import { VoronoiTrackGenerator } from './voronoiTrackGenerator.js';
import { ConvexHullTrackGenerator } from './convexHullTrackGenerator.js';
import * as utils from '../utils/utils.js';
import { JSON_DEBUG } from '../utils/constants.js';
import * as spline from './splineGenerator.js';

let savePointsToJson;


async function importJsonUtils() {
	if (typeof window === 'undefined') {
		const module = await import('../utils/jsonUtils.js');
		savePointsToJson = module.savePointsToJson;
	}
}

export async function generateTrack({ mode, bbox, seed, trackSize, saveJSON = JSON_DEBUG, dataSet = [], selected = [], rngMode, perlin_parameters = null, canonicalize = true } = {}) {
	let trackGenerator;

	if (saveJSON) await importJsonUtils();
	switch (mode) {
		case 'voronoi':
			//in case of Voronoi select -> selected Voronoi cells
			trackGenerator = new VoronoiTrackGenerator(bbox, seed, trackSize, dataSet, selected, rngMode, perlin_parameters);
			break;
		case 'convexHull':
			//in case of convexHull, selected -> selected points from dataset which makes the hull
			trackGenerator = new ConvexHullTrackGenerator(bbox, seed, trackSize, dataSet);
			break;
		default:
			throw new Error('Invalid track generator mode');
	}

	let splineTrack = spline.splineSmoothing(trackGenerator.trackEdges);

	// Canonicalize winding order and start point 
	if (canonicalize) {
		splineTrack = utils.canonicalizeTrack(splineTrack);
	}

	// check if a track has a self-intersection, if so throw an error
	if (utils.hasSelfIntersection(splineTrack)) {
		throw new Error(`Track with seed ${seed} has self-intersection.`);
	}

	let splineVector = utils.resamplePoints(splineTrack); // remanent from older code, can probably be removed
	if (saveJSON) {
		if (mode === 'voronoi')
			await savePointsToJson(seed, trackGenerator.dataSet, mode, rngMode, trackGenerator.selectedCells.map(cell => cell.site), splineVector);
		else
			await savePointsToJson(seed, trackGenerator.dataSet, mode, rngMode, [], splineVector);
	}

	return { track: splineTrack, generator: trackGenerator, splineVector: splineVector };
}