// constants.js
export const BBOX = { xl: 0, xr: 600, yt: 0, yb: 600 };
export const MODE = 'convexHull'; // voronoi or convexHull
export const NUMBER_OF_VORONOI_SITES = 100;
export const MAX_NUMBER_OF_SELECTED_CELLS = 10;
export const DOCKER_IMAGE_NAME = 'torcs:dev';
export const MAPELITE_PATH = './src/utils/mapelite.xml';
export const MEMORY_LIMIT = '2000m';
export const OUTPUT_DIR = './data/voronoi'
export const OUTPUT_DIR_JSON = OUTPUT_DIR + '/json';
export const OUTPUT_DIR_FIT = OUTPUT_DIR + '/fitted';
export const OUTPUT_DIR_XML = OUTPUT_DIR + '/xmlTracks';
export const UTILS_DIR = './src/utils/';
export const LOG_DIR = './logs/';

export const JSON_DEBUG = false;
export const XML_DEBUG = false;
export const SIMULATION_TIMEOUT = 60000;

export const DEFAULT_TRACK_SCALE = 2.0;
export const TARGET_RACE_DURATION = 360; // in seconds
export const DEFAULT_REPETITIONS = 4;
export const COLORS = {
  SEPARATION_LINE: [255, 0, 0],
  VORONOI: [180, 192, 165],    // Sage
  POINTS: [113, 131, 85],     // Reseda Green
  EDGES: [164, 176, 146],     // Sage
  BACKGROUND: [248, 248, 248],  // Light Gray (original background color)
  SPLINE: [
    [247, 37, 133],   // Rose
    [181, 23, 158],   // Fandango
    [114, 9, 183],    // Grape
    [76, 201, 240]    // Vivid Sky Blue
  ]
};

export const RngMode = {
  UNIFORM: 0,
  PERLIN: 1,
};

export const DEFAULT_PERLIN_PARAMETERS = {
  NOISE_FREQUENCY: 3, // Lower = Larger, wider shapes, 3 looks best.
  densityThreshold: 0.3, // Adjusts overall density (0 to 1)
  densityExponent: 2.0, // Adjusts how sharply density falls off (>= 1)
  minDistScale: 0.25 // Scales the minimum distance between Voronoi sites (0.1 to 1, lower = more sites)
};
  
