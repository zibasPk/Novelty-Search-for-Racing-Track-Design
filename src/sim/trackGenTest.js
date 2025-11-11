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
import * as utils from '../utils/utils.js';
import { log } from 'mathjs';


for(let seed = 0; seed < 20000; seed++){
  let result =await generateTrack(
      "voronoi", BBOX, seed, (seed % 8) + 1,
      true, [], []
  );
  const trackXml = xml.exportTrackToXML(result.track, 0, true, seed);
}

// let current = {x: -1, y: 5};
// let startOfStraight = {x: 2, y: 2};

// let straightHeadingVector = utils.normalizeVector({ x: current.x - startOfStraight.x, y: current.y - startOfStraight.y });

// // get and remove the last curve section added
// const prevCurve = {
//   radius: 2,
//   center: {x:2, y:0},
//   dir: "lft"
// };
// // recalculate the curve last point to account for the straight segment
// startOfStraight = utils.getPointFromHeading(straightHeadingVector, prevCurve.radius, prevCurve.center, prevCurve.dir);
// // check new straight heading
// let newStraightHeadingVector = utils.normalizeVector({
//   x: current.x - startOfStraight.x,
//   y: current.y - startOfStraight.y
// });

// // calculate the new curve angle based on the new point
// prevCurve.angle = utils.calculateAngle(prevCurve.center, startOfStraight, prevCurve.dir);

// let curveheading = utils.calculateCurveHeading(prevCurve.angle, prevCurve.dir, straightHeadingVector);

// console.log('New Curve Angle:', newCurve.angle);




 





