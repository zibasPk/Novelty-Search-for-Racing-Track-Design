import hermite from "cubic-hermite";
import log from "loglevel";

export function splineSmoothing(spline) {
  for (let i = 0; i < 10; i++) {
    spline = generateCatmullRomSpline(spline, 5, i * 10);
    spline = pushApart(spline, 20);
    spline = fixAngles(spline);
  }
  spline = generateCatmullRomSpline(spline, 20, 0);
  spline = pushApart(spline, 0.1);
  return spline;
}

export function splineSmoothingWithStraights(spline, minStraightLength = 10, maxCurvature = 0.01) {

  let straightenSegment = (spline, startIndex, segmentLength) => {
    const trackLength = spline.length;
    const endIndex = (startIndex + segmentLength) % trackLength;

    // Calculate the direction vector for the straight segment
    const startPoint = spline[startIndex];
    const endPoint = spline[endIndex];
    const dx = (endPoint.x - startPoint.x) / segmentLength;
    const dy = (endPoint.y - startPoint.y) / segmentLength;

    // Adjust points in the segment to align linearly
    for (let i = 0; i < segmentLength; i++) {
      const index = (startIndex + i) % trackLength;
      spline[index] = {
        x: startPoint.x + i * dx,
        y: startPoint.y + i * dy
      };
    }

    return spline;
  }

  let smoothSpline = splineSmoothing(spline);
  let resultSpline = smoothSpline;
  let segment = {
    start: 0,
    end: 2,
    length: function (trackLength) {
      if (this.end >= this.start) {
        return this.end - this.start + 1;
      }
      return trackLength - this.start + this.end + 1;
    }
  };

  let totalCurvature = 0;
  let firstStraightIdx = -1;
  let straightenedSegments = 0;

  let i = 0;
  let trackLength = smoothSpline.length;
  let loopEnd = trackLength - 2;
  while (i < loopEnd) {
    const index = i % trackLength;
    const curvature = utils.calculateCurvature(smoothSpline, index);
    totalCurvature += curvature;
    segment.end = (index + 2) % trackLength;

    if (totalCurvature / segment.length(trackLength) > maxCurvature) {
      const prevSegLen = segment.length(trackLength) - 1;
      if (prevSegLen > minStraightLength) {
        // straighten segment up to previous point
        resultSpline = straightenSegment(resultSpline, segment.start, prevSegLen);
        straightenedSegments++;
        log.debug("straightened a segment of length: ", prevSegLen);
        if (firstStraightIdx === -1) {
          // store first straight segment index for later adjustments
          firstStraightIdx = segment.start;
          loopEnd = trackLength + firstStraightIdx;
        }
        // reset segment
        segment.start = segment.end;
      } else {
        segment.start = index;
      }
      totalCurvature = 0;
    }
    i++;
  }
  log.debug("Total straightened segments: ", straightenedSegments);

  return resultSpline;
}

/**
 * Push apart points that are too close to each other.
 * @param {*} points 
 * @param {*} minDistance 
 * @returns 
 */
export function pushApart(points, minDistance = 5) {
  const minDistanceSquared = minDistance * minDistance;
  let i = 0;
  while (i < points.length) {
    let removed = false;
    for (let j = i + 1; j < points.length; j++) {
      const dx = points[j].x - points[i].x;
      const dy = points[j].y - points[i].y;
      const distanceSquared = dx * dx + dy * dy;
      if (distanceSquared < minDistanceSquared) {
        points.splice(j, 1);
        removed = true;
        break;
      }
    }
    if (!removed) {
      i++;
    }
  }
  return points;
}
/**
 * Fix angles that are too sharp by limiting the angle between consecutive segments.
 * @param {*} points 
 * @returns 
 */
export function fixAngles(points) {
  const radDeg = 180 / Math.PI;
  const degRad = Math.PI / 180;
  const maxAngle = 80;

  for (let i = 0; i < points.length; ++i) {
    const previous = (i - 1 < 0) ? points.length - 1 : i - 1;
    const next = (i + 1) % points.length;

    let px = points[i].x - points[previous].x;
    let py = points[i].y - points[previous].y;
    const pl = Math.sqrt(px * px + py * py);
    px /= pl;
    py /= pl;

    let nx = points[next].x - points[i].x;
    let ny = points[next].y - points[i].y;
    const nl = Math.sqrt(nx * nx + ny * ny);
    nx /= nl;
    ny /= nl;

    let a = Math.atan2(px * ny - py * nx, px * nx + py * ny);
    if (Math.abs(a * radDeg) <= maxAngle) continue;

    const nA = maxAngle * Math.sign(a) * degRad;
    const diff = nA - a;
    const cos = Math.cos(diff);
    const sin = Math.sin(diff);

    const newX = nx * cos - ny * sin;
    const newY = nx * sin + ny * cos;
    points[next].x = points[i].x + newX * nl;
    points[next].y = points[i].y + newY * nl;
  }

  return points;
}

export function generateCatmullRomSpline(data, steps, startIndex) {
  let spline = [];

  for (let i = startIndex; i < startIndex + data.length; i++) {
    const index = i % data.length;
    const p0 = data[(index + data.length - 1) % data.length];
    const p1 = data[index];
    const p2 = data[(index + 1) % data.length];
    const p3 = data[(index + 2) % data.length];

    for (let t = 0; t <= 1; t += 1 / steps) {
      const t2 = t * t;
      const t3 = t2 * t;

      const b1 = 0.5 * (-t3 + 2 * t2 - t);
      const b2 = 0.5 * (3 * t3 - 5 * t2 + 2);
      const b3 = 0.5 * (-3 * t3 + 4 * t2 + t);
      const b4 = 0.5 * (t3 - t2);

      const x = p0.x * b1 + p1.x * b2 + p2.x * b3 + p3.x * b4;
      const y = p0.y * b1 + p1.y * b2 + p2.y * b3 + p3.y * b4;

      spline.push({ x, y });
    }
  }

  return spline;
}