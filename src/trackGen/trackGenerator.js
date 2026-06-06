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

export async function generateTrack({ mode, bbox, seed, trackSize, saveJSON = JSON_DEBUG, dataSet = [], selected = [], rngMode, perlin_parameters = null } = {}) {
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

	// Canonicalize winding order, note: signedArea isn't consistent if we allow self-intersecting tracks but we check for that later and throw an error, so  it should be fine
	if (utils.signedArea(splineTrack) > 0) {
		// Find track center to mirror it in place
		let minX = Infinity, maxX = -Infinity;
		for (let p of splineTrack) {
			if (p.x < minX) minX = p.x;
			if (p.x > maxX) maxX = p.x;
		}
		let centerX = (minX + maxX) / 2;

		// Physically flip all the curves (Right turns become Left turns)
		for (let p of splineTrack) {
			p.x = 2 * centerX - p.x; 
		}
	}

	const minIndex2 = utils.findLongestStraightSegment(splineTrack, 0.01, 0.5);
	let splineTrack2 = splineTrack.slice(minIndex2).concat(splineTrack.slice(0, minIndex2));
	splineTrack = splineTrack2;

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