
import { BBOX } from "./constants.js";
import log from "loglevel";

/**
 * Find the segment with the minimum curvature in the track.
 * @param {*} track 
 * @param {*} segmentLength 
 * @returns The index of the middle point of the segment with minimum curvature. 
 */
export function findMinCurvatureSegment(track, segmentLength) {
  let minAverageCurvature = Infinity;
  let minSegmentStartIndex = 0;
  const trackLength = track.length;

  for (let index = 0; index < trackLength; index++) {
    let totalCurvature = 0;

    for (let offset = 0; offset < segmentLength; offset++) {
      const curvatureIndex = (index + offset) % trackLength;
      const curvature = calculateCurvature(track, curvatureIndex);
      totalCurvature += curvature;
    }

    const averageCurvature = totalCurvature / segmentLength;
    if (averageCurvature < minAverageCurvature) {
      minAverageCurvature = averageCurvature;
      minSegmentStartIndex = index;
    }
  }

  return (minSegmentStartIndex + segmentLength / 2) % trackLength;
}

export function findMaxCurveBeforeStraight(track, segmentLength) {
  let bestScore = -Infinity;
  let desiredSegmentStartIndex = 0;
  const trackLength = track.length;
  const halfSegmentLength = Math.floor(segmentLength / 2);

  for (let index = 0; index < trackLength; index++) {
    let firstHalfTotalCurvature = 0;
    let secondHalfTotalCurvature = 0;

    // Calculate total curvature for the first half of the segment
    for (let offset = 0; offset < halfSegmentLength; offset++) {
      const curvatureIndex = (index + offset) % trackLength;
      const curvature = calculateCurvature(track, curvatureIndex);
      firstHalfTotalCurvature += curvature;
    }

    // Calculate total curvature for the second half of the segment
    for (let offset = halfSegmentLength; offset < segmentLength; offset++) {
      const curvatureIndex = (index + offset) % trackLength;
      const curvature = calculateCurvature(track, curvatureIndex);
      secondHalfTotalCurvature += curvature;
    }

    const firstHalfAverageCurvature = firstHalfTotalCurvature / halfSegmentLength;
    const secondHalfAverageCurvature = secondHalfTotalCurvature / halfSegmentLength;

    // Score calculation
    const curvatureScore = firstHalfAverageCurvature; // Higher score for higher curvature
    const straightnessScore = 1 / (secondHalfAverageCurvature + 1); // Higher score for lower curvature (more straight)

    const totalScore = straightnessScore + curvatureScore;

    // Update if current segment has a better score
    if (totalScore > bestScore) {
      bestScore = totalScore;
      desiredSegmentStartIndex = index;
    }
  }

  return (desiredSegmentStartIndex + segmentLength / 2) % trackLength;
}

/**
 * Return an index of the longest straight segment in the track.
 * "Longest" is defined by physical distance, not number of points.
 * 
 * @param {Array<{x: number, y: number}>} track - Array of point objects
 * @param {number} curvatureThreshold - Max deviation in radians (0 = perfectly straight)
 * @param {number} indexPos - number between 0 and 1 to offset the returned index within the segment (0=start, 0.5=middle, 1=end)
 * @returns {number} The index of the middle point of the longest straight section
 */
export function findLongestStraightSegment(track, curvatureThreshold, indexPos = 0.5) {
  // Edge cases: not enough points to form a segment
  if (!track || track.length < 2) return 0;

  let maxDistance = 0;
  let longestSegmentMidIndex = 0;

  let currentStartIndex = 0;
  let currentDistance = 0;

  // Helper to calculate distance between two points
  const getDist = (p1, p2) => Math.hypot(p2.x - p1.x, p2.y - p1.y);

  // Helper to calculate angle deviation (0 to PI)
  const getDeviation = (p1, p2, p3) => {
    // Vector 1 (p1 -> p2)
    const dx1 = p2.x - p1.x;
    const dy1 = p2.y - p1.y;
    // Vector 2 (p2 -> p3)
    const dx2 = p3.x - p2.x;
    const dy2 = p3.y - p2.y;

    // Calculate angles
    const angle1 = Math.atan2(dy1, dx1);
    const angle2 = Math.atan2(dy2, dx2);

    // Calculate difference
    let diff = Math.abs(angle1 - angle2);

    // Normalize to [0, PI] (handle wrap-around, e.g. 359° vs 1°)
    if (diff > Math.PI) {
      diff = 2 * Math.PI - diff;
    }
    
    return diff;
  };

  // Iterate through the track
  // We start at 1 because we need a previous point to measure a vector
  for (let i = 1; i < track.length; i++) {
    // Add the distance of the latest step (i-1 to i) to current tally
    currentDistance += getDist(track[i - 1], track[i]);

    // Check curvature
    // We can only check curvature if we have a 'next' point (i+1)
    let isCurveTooSharp = false;

    if (i < track.length - 1) {
      const deviation = getDeviation(track[i - 1], track[i], track[i + 1]);
      if (deviation > curvatureThreshold) {
        isCurveTooSharp = true;
      }
    }

    // If the curve is too sharp, or we are at the very end of the array
    if (isCurveTooSharp || i === track.length - 1) {
      // Check if this run was the longest found so far
      if (currentDistance > maxDistance) {
        maxDistance = currentDistance;
        // Calculate middle index between start of this segment and current point
        
        longestSegmentMidIndex = Math.floor((currentStartIndex + i) * indexPos);
      }

      // Reset for next segment
      // The new segment technically starts at the point where the curve happened (i)
      currentStartIndex = i;
      currentDistance = 0;
    }
  }

  return longestSegmentMidIndex;
}

/**
 * Calculate curvature of a sequence of three points in the track starting at index i.
 * @param {*} track 
 * @param {*} i 
 * @param {*} absolute If true, return absolute curvature; if false, return signed curvature.
 * @returns 
 */
export function calculateCurvature(track, i, absolute = true) {
  const current = track[i];
  const next = track[(i + 1) % track.length];
  const nextNext = track[(i + 2) % track.length];

  const xp = (nextNext.x - current.x) / 2;
  const yp = (nextNext.y - current.y) / 2;
  const xpp = (nextNext.x - 2 * next.x + current.x);
  const ypp = (nextNext.y - 2 * next.y + current.y);

  const numerator = xp * ypp - yp * xpp;
  const denominator = Math.pow((xp * xp + yp * yp), 1.5);

  if (absolute) {
    return denominator !== 0 ? Math.abs(numerator / denominator) : 0;
  } else {
    return denominator !== 0 ? numerator / denominator : 0;
  }
}

export function calculateSegment(point1, point2) {
  const dx = point2.x - point1.x;
  const dy = point2.y - point1.y;
  return Math.sqrt(dx * dx + dy * dy);
}

export function calculateCurve(p1, p2, p3) {
  function determinant(x1, y1, x2, y2, x3, y3) {
    return x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2);
  }

  const D = 2 * determinant(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);

  if (D === 0) return null;

  const ux = ((p1.x ** 2 + p1.y ** 2) * (p2.y - p3.y) +
    (p2.x ** 2 + p2.y ** 2) * (p3.y - p1.y) +
    (p3.x ** 2 + p3.y ** 2) * (p1.y - p2.y)) / D;
  const uy = ((p1.x ** 2 + p1.y ** 2) * (p3.x - p2.x) +
    (p2.x ** 2 + p2.y ** 2) * (p1.x - p3.x) +
    (p3.x ** 2 + p3.y ** 2) * (p2.x - p1.x)) / D;
  const radius = Math.sqrt((p1.x - ux) ** 2 + (p1.y - uy) ** 2);

  const angle1 = Math.atan2(p1.y - uy, p1.x - ux);
  const angle3 = Math.atan2(p3.y - uy, p3.x - ux);

  let theta = Math.abs(angle3 - angle1) * (180 / Math.PI);
  if (theta > 180) theta = 360 - theta;

  const dir = (p2.x - p1.x) * (p3.y - p2.y) - (p2.y - p1.y) * (p3.x - p2.x) > 0 ? 'rgt' : 'lft';

  return { dir, radius, angle: theta, center: { x: ux, y: uy } };
}
/**
 * Calculate new heading vector after a turn.
 * @param {*} turnAngle Turning angle in degrees
 * @param {*} direction 'rgt' for right, 'lft' for left
 * @param {*} initialHeading normalized heading before the curve with x and y components
 * @returns new normalized heading vector with x and y components
 */
export function calculateCurveHeading(turnAngle, direction, initialHeading) {
  turnAngle = direction === 'lft' ? turnAngle : -turnAngle;
  const theta = turnAngle * (Math.PI / 180); // Convert to radians
  const cosTheta = Math.cos(theta);
  const sinTheta = Math.sin(theta);

  const newX = initialHeading.x * cosTheta - initialHeading.y * sinTheta;
  const newY = initialHeading.x * sinTheta + initialHeading.y * cosTheta;

  // normalize the new heading vector
  return normalizeVector({ x: newX, y: newY });
}

export function getCircleCenter(radius, p1, p2, direction) {
  // Midpoint of chord
  const M = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };

  // Length of chord
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const L = Math.hypot(dx, dy);

  if (L / 2 > radius) {
    throw new Error("Radius is too small for the given points.");
  }

  // Distance from midpoint to circle center
  const h = Math.sqrt(radius * radius - (L / 2) * (L / 2));

  // Unit normal to chord (points to the left of direction p1 -> p2)
  const nx = -dy / L;
  const ny = dx / L;

  // Center of circle (depends on turn direction)
  const sign = direction === "lft" ? 1 : -1;
  const C = { x: M.x + sign * h * nx, y: M.y + sign * h * ny };

  return C;
}

export function normalizeAngle(angle) {
  // Bring the angle within (-π, π]
  angle = Math.atan2(Math.sin(angle), Math.cos(angle));
  return angle;
};

/**
 * Calculate a heading vector at point p1 for a curve to p2 with given radius and direction.
 * @param {*} p1 start point of the curve
 * @param {*} p2 end point of the curve
 * @param {*} radius radius of the curve
 * @param {*} direction 'rgt' for right, 'lft' for left
 * @returns new normalized heading vector with x and y components
 */
export function calculateCurveInitialHeading(p1, p2, radius, direction) {
  const C = getCircleCenter(radius, p1, p2, direction);
  // Radius vector at p1
  const rx = p1.x - C.x;
  const ry = p1.y - C.y;

  // Tangent vector: perpendicular to radius
  let tx, ty;
  if (direction === "lft") {
    tx = -ry;
    ty = rx;
  } else {
    tx = ry;
    ty = -rx;
  }

  // Normalize tangent
  return normalizeVector({ x: tx, y: ty });
}

export function normalizeVector(vector) {
  const length = Math.hypot(vector.x, vector.y);
  return { x: vector.x / length, y: vector.y / length };
}

/**
 * Check whether two line segments (p1→p2) and (p3→p4) intersect.
 * Uses the standard cross-product orientation test.
 * Returns true if segments properly cross each other (not just touch at endpoints).
 */
function segmentsIntersect(p1, p2, p3, p4) {
  const cross = (o, a, b) =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const d1 = cross(p3, p4, p1);
  const d2 = cross(p3, p4, p2);
  const d3 = cross(p1, p2, p3);
  const d4 = cross(p1, p2, p4);

  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
      ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) {
    return true;
  }

  // Collinear / on-segment cases (touch)
  const onSegment = (p, q, r) =>
    Math.min(p.x, r.x) <= q.x && q.x <= Math.max(p.x, r.x) &&
    Math.min(p.y, r.y) <= q.y && q.y <= Math.max(p.y, r.y);

  if (d1 === 0 && onSegment(p3, p1, p4)) return true;
  if (d2 === 0 && onSegment(p3, p2, p4)) return true;
  if (d3 === 0 && onSegment(p1, p3, p2)) return true;
  if (d4 === 0 && onSegment(p1, p4, p2)) return true;

  return false;
}

/**
/**
 * Check if a closed track (array of {x, y} points) has a self-intersection.
 * Compares every pair of non-adjacent segments. Adjacent segments naturally
 * share an endpoint and are skipped. For the closing segment (last→first),
 * adjacency with the first and last segment is also handled.
 *
 * If the last point is (nearly) identical to the first point the track is
 * already closed — the duplicate tail is stripped so that the implicit
 * closing segment does not create a degenerate zero-length edge that would
 * overlap with its neighbours and trigger false positives.
 *
 * @param {Array<{x: number, y: number}>} track
 * @param {number} [epsilon=1e-9] tolerance for duplicate first/last point
 * @returns {boolean} true if any two non-adjacent segments cross
 */
export function hasSelfIntersection(track, epsilon = 1e-9) {
  if (!track || track.length < 4) return false;

  // Strip duplicate closing point(s) so the implicit wrap-around segment
  // (last → first) is never degenerate / overlapping.
  let pts = track;
  while (
    pts.length > 3 &&
    Math.abs(pts[pts.length - 1].x - pts[0].x) < epsilon &&
    Math.abs(pts[pts.length - 1].y - pts[0].y) < epsilon
  ) {
    pts = pts.slice(0, -1);
  }

  const n = pts.length;

  for (let i = 0; i < n; i++) {
    const a1 = pts[i];
    const a2 = pts[(i + 1) % n];

    // Start j at i + 2 so we skip the immediately adjacent segment
    for (let j = i + 2; j < n; j++) {
      // Skip the pair where the closing segment wraps around to segment 0
      if (i === 0 && j === n - 1) continue;

      const b1 = pts[j];
      const b2 = pts[(j + 1) % n];

      if (segmentsIntersect(a1, a2, b1, b2)) {
        return true;
      }
    }
  }

  return false;
}

export function resamplePoints(points) {
  // Resample the normalized track into a fixed-length vector (e.g., 100 points).
  const numSamples = 100;

  if (points.length > 0) {
    const first = points[0];
    const last = points[points.length - 1];
    if (first.x !== last.x || first.y !== last.y) {
      points.push({ x: first.x, y: first.y });
    }
  }

  // Normalize the track points into the unit square.
  const normalizedPoints = points.map(pt => ({
    x: pt.x / BBOX.xr,
    y: pt.y / BBOX.yb
  }));

  if (normalizedPoints.length < 2) return normalizedPoints;

  // Compute cumulative distances along the polyline.
  const distances = [0];
  for (let i = 1; i < normalizedPoints.length; i++) {
    const dx = normalizedPoints[i].x - normalizedPoints[i - 1].x;
    const dy = normalizedPoints[i].y - normalizedPoints[i - 1].y;
    distances.push(distances[i - 1] + Math.hypot(dx, dy));
  }

  const totalLength = distances[distances.length - 1];
  let resampled = [];

  // For each sample point, compute its target distance and interpolate.
  for (let i = 0; i < numSamples; i++) {
    const target = (i / (numSamples - 1)) * totalLength;
    let j = 1;
    while (j < distances.length && distances[j] < target) {
      j++;
    }
    const t = (target - distances[j - 1]) / (distances[j] - distances[j - 1]);
    const x = normalizedPoints[j - 1].x + t * (normalizedPoints[j].x - normalizedPoints[j - 1].x);
    const y = normalizedPoints[j - 1].y + t * (normalizedPoints[j].y - normalizedPoints[j - 1].y);
    resampled.push({ x, y });
  }

  // Centring (translate so centroid == (0,0))
  const meanX = resampled.reduce((sum, p) => sum + p.x, 0) / resampled.length;
  const meanY = resampled.reduce((sum, p) => sum + p.y, 0) / resampled.length;

  resampled = resampled.map(p => ({
    x: p.x - meanX,
    y: p.y - meanY
  }));
  return resampled;
}

