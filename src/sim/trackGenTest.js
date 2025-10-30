import {
  BBOX,
  MODE,
  DOCKER_IMAGE_NAME,
  MEMORY_LIMIT,
  SIMULATION_TIMEOUT,
  OUTPUT_DIR_XML
} from '../utils/constants.js';
import { generateTrack } from '../trackGen/trackGenerator.js';
import * as xml from '../utils/xmlTorcsGenerator.js';
import { createCanvas, circle, background } from 'p5';

let result =await generateTrack(
    "voronoi", BBOX, 1725, (1725 % 8) + 1,
    true, [], []
);

const trackXml = xml.exportTrackToXML(result.track, 0, true, 1725);





