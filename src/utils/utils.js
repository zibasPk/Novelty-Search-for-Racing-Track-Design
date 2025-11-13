import { BBOX } from "./constants.js";
import log from "loglevel"





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
 * Calculate curvature of a sequence of three points in the track starting at index i.
 * @param {*} track 
 * @param {*} i 
 * @returns 
 */
export function calculateCurvature(track, i) {
  const current = track[i];
  const next = track[(i + 1) % track.length];
  const nextNext = track[(i + 2) % track.length];

  const xp = (nextNext.x - current.x) / 2;
  const yp = (nextNext.y - current.y) / 2;
  const xpp = (nextNext.x - 2 * next.x + current.x);
  const ypp = (nextNext.y - 2 * next.y + current.y);

  const numerator = xp * ypp - yp * xpp;
  const denominator = Math.pow((xp * xp + yp * yp), 1.5);

  return denominator !== 0 ? Math.abs(numerator / denominator) : 0;
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

export function calculateFinalPose(sections, startPose = { x: 0, y: 0, heading: 0 }) {
  let pose = { ...startPose };

  for (const s of sections) {
    if (s.type === 'straight') {
      // move forward
      pose.x += s.length * Math.cos(pose.heading);
      pose.y += s.length * Math.sin(pose.heading);
      // heading unchanged
    }
    else if (s.type === 'curve') {
      const dirSign = s.dir === 'lft' ? 1 : -1;
      const angRad = (s.angle * Math.PI) / 180;

      // find circle center
      const cx = pose.x - dirSign * s.radius * Math.sin(pose.heading);
      const cy = pose.y + dirSign * s.radius * Math.cos(pose.heading);

      // advance heading
      pose.heading += dirSign * angRad;

      // new position on circle
      pose.x = cx + dirSign * s.radius * Math.sin(pose.heading);
      pose.y = cy - dirSign * s.radius * Math.cos(pose.heading);
    }
  }

  // normalize heading to [-PI, PI]
  pose.heading = ((pose.heading + Math.PI) % (2 * Math.PI)) - Math.PI;

  return pose;
}
