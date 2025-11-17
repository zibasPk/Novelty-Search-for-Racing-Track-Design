import {
  BBOX,
} from '../utils/constants.js';
import { generateTrack } from '../trackGen/trackGenerator.js';
import {TorcsXMLGenerator} from '../utils/torcsXMLGenerator.js';
import * as xml from '../utils/xmlTorcsGenerator.js';
import log from "loglevel"


let startSeed = 0;
const trackLengths = [];

for(let seed = startSeed; seed < startSeed + 4000; seed++){
  log.setLevel("warn");
  let result =await generateTrack(
      "voronoi", BBOX, seed, (seed % 8) + 1,
      true, [], []
  );
  trackLengths.push(result.track.length);
  log.setLevel("debug");
  try {
  // const trackXml = xml.exportTrackToXML(result.track, 0, true, seed);
  let xmlGenerator = new TorcsXMLGenerator(result.track, seed);
  xmlGenerator.generateXML(0, true);
  } catch (e) {
    log.error(`Error generating xml for track seed ${seed} : ${e.message}`);
  }
}

// get some statistics on track lengths
const sum = trackLengths.reduce((a, b) => a + b, 0);
const avg = sum / trackLengths.length;
//get min and max and their indexes
let max = Math.max(...trackLengths);

// find top 10 minimum lengths and indexes
const top10MinLengths = trackLengths
  .map((length, index) => ({ length, index }))
  .sort((a, b) => a.length - b.length)
  .slice(0, 10);

log.info("Top 10 Minimum Track Lengths:");
top10MinLengths.forEach(({ length, index }) => {
  log.info(`Index: ${index}, Length: ${length}`);
});




log.info(`Track Lengths Statistics over ${trackLengths.length} tracks:`);
log.info(`Average: ${avg}`);
log.info(`Max: ${max}`);
// additonal statisics
const sortedLengths = trackLengths.slice().sort((a, b) => a - b);
const median = (sortedLengths.length % 2 === 0) ?
  (sortedLengths[sortedLengths.length / 2 - 1] + sortedLengths[sortedLengths.length / 2]) / 2 :
  sortedLengths[Math.floor(sortedLengths.length / 2)];
log.info(`Median: ${median}`);

const variance = trackLengths.reduce((a, b) => a + Math.pow(b - avg, 2), 0) / trackLengths.length;
const stdDev = Math.sqrt(variance);
log.info(`Standard Deviation: ${stdDev}`);

// test exportTrackToXML function with 7 points
// let track;
// track = [
//   {x: 0, y: 0},
//   {x: 5, y: 0},
//   {x: 7, y: 2},
//   {x: 5, y: 5},
//   {x: 0, y: 5},
//   {x: -2, y: 2},
//   {x: -1, y: 1},
// ];

// xml.exportTrackToXML(track, 0, true, 'test7points');
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




 





