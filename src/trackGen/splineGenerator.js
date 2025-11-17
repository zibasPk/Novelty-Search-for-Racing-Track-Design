import hermite from "cubic-hermite";
import log from "loglevel";

export function cubicSplineSmoothing(spline, inTangent, outTangent) {
  // Initial smoothing passes with fewer subdivisions
  for (let i = 0; i < 10; i++) {
    spline = cubicHermiteSpline(spline, null, null, 5);
    spline = pushApart(spline, 20);
    spline = fixAngles(spline);
  }
  
  // Final pass with more subdivisions
  spline = cubicHermiteSpline(spline, inTangent, outTangent, 20);
  spline = pushApart(spline, 0.1);
  
  return spline;
}

function clampTangent(tangent, maxMagnitude) {
  const magnitude = Math.sqrt(tangent[0] ** 2 + tangent[1] ** 2);
  
  if (magnitude > maxMagnitude) {
    const scale = maxMagnitude / magnitude;
    return [tangent[0] * scale, tangent[1] * scale];
  }
  
  return tangent;
}

function calculateTangents(points, maxTangentLength = null) {
  return points.map((point, i) => {
    let tangent;
    
    if (i === 0) {
      tangent = [points[1].x - point.x, points[1].y - point.y];
    } else if (i === points.length - 1) {
      tangent = [point.x - points[i - 1].x, point.y - points[i - 1].y];
    } else {
      tangent = [
        (points[i + 1].x - points[i - 1].x) * 0.5,
        (points[i + 1].y - points[i - 1].y) * 0.5
      ];
    }
    
    // Clamp tangent if max length is specified
    return maxTangentLength !== null 
      ? clampTangent(tangent, maxTangentLength) 
      : tangent;
  });
}

function getSegmentCount(p1, p2, segmentsPerCurve) {
  if (segmentsPerCurve !== null) return segmentsPerCurve;
  
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const distance = Math.sqrt(dx * dx + dy * dy);
  return Math.max(2, Math.floor(distance / 2));
}

export function cubicHermiteSpline(
  points, 
  inTangent = null, 
  outTangent = null, 
  segmentsPerCurve = null,
  maxTangentLength = null  
) {
  const tangents = calculateTangents(points, maxTangentLength);
  
  if (inTangent) tangents[0] = inTangent;
  if (outTangent) tangents[tangents.length - 1] = outTangent;
  
  // Override first and last tangents if provided
  if (inTangent) tangents[0] = inTangent;
  if (outTangent) tangents[tangents.length - 1] = outTangent;
  
  const allPoints = [];
  
  for (let i = 0; i < points.length - 1; i++) {
    const segments = getSegmentCount(points[i], points[i + 1], segmentsPerCurve);
    
    for (let j = 0; j <= segments; j++) {
      const t = j / segments;
      const point = [0, 0];
      
      hermite(
        [points[i].x, points[i].y],
        tangents[i],
        [points[i + 1].x, points[i + 1].y],
        tangents[i + 1],
        t,
        point
      );
      
      allPoints.push({ x: point[0], y: point[1] });
    }
  }
  
  return allPoints;
}

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