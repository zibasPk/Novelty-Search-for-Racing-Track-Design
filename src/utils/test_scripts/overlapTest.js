import * as math from 'mathjs';


let overLappedResults = [];
/**
 * Checks if a closed loop path overlaps itself given a specific width.
 * @param {Array<{x: number, y: number}>} track - Array of points {x, y}
 * @param {number} trackWidth - The total width of the path
 * @returns {boolean}
 */
function trackHasSelfOverlap(track, trackWidth, trackId) {
  const len = track.length;
  for (let i = 0; i < len; i++) {
    const a1 = track[i];
    const a2 = track[(i + 1) % len];
    let a_direction = { x: a2.x - a1.x, y: a2.y - a1.y };
    for (let j = i + 2; j < len; j++) {
      if (i === 0 && j === len - 1) continue;

      const b1 = track[j];
      const b2 = track[(j + 1) % len];

      // check if the intersect
      if (segmentsIntersect(a1, a2, b1, b2)) {
        return true;
      }

      let b_direction = { x: b2.x - b1.x, y: b2.y - b1.y };
      let theta = smallestAngle2D(a_direction, b_direction);

      if (theta < Math.PI / 2) continue;

      let distance = segToSegMinDistance(a1, a2, b1, b2);
      if (distance < trackWidth) {
        overLappedResults.push(trackId);
        return true;
      }
    }
  }
  return false;
}

function smallestAngle2D(a, b) {
  const dot = a.x * b.x + a.y * b.y;
  const magA = Math.hypot(a.x, a.y);
  const magB = Math.hypot(b.x, b.y);

  if (magA === 0 || magB === 0) return 0;

  const cosTheta = Math.max(-1, Math.min(1, dot / (magA * magB)));
  return Math.acos(cosTheta); // in radians range [0, π]
}

function segToSegMinDistance(a1, a2, b1, b2) {
  

  // 2. If no intersection, distance is the min distance of the 4 endpoints
  const d1 = pointToSegmentDistance(a1, b1, b2);
  const d2 = pointToSegmentDistance(a2, b1, b2);
  const d3 = pointToSegmentDistance(b1, a1, a2);
  const d4 = pointToSegmentDistance(b2, a1, a2);

  return Math.min(d1, d2, d3, d4);
}

function pointToSegmentDistance(P, A, B) {
  const dx = B.x - A.x;
  const dy = B.y - A.y;

  const lenSq = dx * dx + dy * dy;

  // Handle zero-length segments to avoid NaN
  if (lenSq === 0) {
    return Math.hypot(P.x - A.x, P.y - A.y);
  }

  const t = ((P.x - A.x) * dx + (P.y - A.y) * dy) / lenSq;

  // Clamp t to segment [0, 1]
  const tClamped = Math.max(0, Math.min(1, t));

  const closestX = A.x + tClamped * dx;
  const closestY = A.y + tClamped * dy;

  return Math.hypot(P.x - closestX, P.y - closestY);
}

function segmentsIntersect(p1, p2, q1, q2) {
  // Standard Cross Product (Z-component)
  const cross = (a, b, c) => (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);

  const d1 = cross(p1, p2, q1);
  const d2 = cross(p1, p2, q2);
  const d3 = cross(q1, q2, p1);
  const d4 = cross(q1, q2, p2);

  // Using strict inequality (< 0) handles clean intersections. 
  // If d is 0, points are collinear/touching, which pointToSegment handles better.
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
    ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
    return true;
  }

  return false;
}

async function genTrackAndCheckOverlap(trackData) {
  const trackResults = await generateTrack(
    trackData.mode, BBOX, trackData.id, trackData.selectedCells.length,
    false, trackData.dataSet, trackData.selectedCells
  );

  if (trackHasSelfOverlap(trackResults.track, 10, trackData.id)) {
    return trackData.id;
  }

  return null;
}


// run the test over all json in the fitted directory
import fs from 'fs/promises';
import log from "loglevel"
import { generateTrack } from '../../trackGen/trackGenerator.js';
import {
  BBOX,
} from '../constants.js';

const fittedDir = 'data/voronoi/fitted/';
const files = await fs.readdir(fittedDir);
log.setLevel("info");

let results = [];
let fileCount = 0;
let justOne = false; // set to true to test just one file
let startTime = Date.now();
for (const file of files) { 
  if (file.endsWith('.json')) {
    fileCount++;
    try {
      const filePath = fittedDir + file;
      const data = await fs.readFile(filePath, 'utf-8');
      const trackData = JSON.parse(data);
      let res = await genTrackAndCheckOverlap(trackData);
      if (res !== null) {
        results.push(res);
      }
    } catch (e) {
      log.error(`Error processing file ${file}: ${e.message}`);
    }
    if (justOne) break;
  }
  // if (Date.now() - startTime > 30000) {
  //   log.info("Timeout reached, stopping further processing.");
  //   break;
  // }
}
log.info("Overlap Test Results:", results);
log.info(`Processed ${fileCount} files.`);
log.info(`Total tracks with self-overlap: ${results.length}`);
log.info("Overlapped Track IDs:", overLappedResults);