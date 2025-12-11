import fs from 'fs';
import path from 'path';
import * as utils from '../utils/utils.js';
import { OUTPUT_DIR_XML, UTILS_DIR, DEFAULT_TRACK_SCALE} from '../utils/constants.js';
import log from "loglevel";
import { PositionCorrectionError } from "../utils/errors.js";



export class TorcsXMLGenerator {
  constructor(track, seed = 0, scale = DEFAULT_TRACK_SCALE) {
    this.track = track;
    this.seed = seed;

    this.xmlTrackHeader = fs.readFileSync(path.join(UTILS_DIR, 'startTrackTemplate.xml'), 'utf8');
    this.closingXML = "</section>\n</section>\n</params>";
    this.xml = '';
    this.sections = [];
    this.logHeader = `Track ${seed}: `;
    this.trackScale = scale; // scale factor for track dimensions
  }

  /**
 * Generates an XML representation of a racing track based on the point coordinates.
 * The function identifies straight and curved sections of the track, calculates their parameters,
 * and constructs the corresponding XML structure. It also checks and adds corrections for track closure errors
 * @param {*} startIndex 
 * @param {*} saveXMLalsoLocally 
 * @returns 
 */
  generateXML(startIndex = 0, saveXMLalsoLocally = false) {
    const threshold = 0.001;
    const sections = this.sections;
    const track = this.track;
    const trackName = this.seed;

    let startOfStraightIdx = null;
    let endOfStraightIdx = null;
    let nullCurveCounter = 0;

    let index = startIndex;
    while (index < startIndex + track.length - 1) {
      const i = (index) % track.length; const i_next = (index + 1) % track.length; const i_nextnext = (index + 2) % track.length;
      const current = track[i]; const next = track[i_next]; const nextNext = track[i_nextnext];


      const curvature = utils.calculateCurvature(track, i);
      if (curvature < threshold) {
        if (startOfStraightIdx === null) {
          startOfStraightIdx = i;
        }
        if (index >= startIndex + track.length - 3) {
          endOfStraightIdx = i_nextnext;
        }
        index++; // skip a point since we used nextNext
      } else {
        // found a curve after straights
        if (startOfStraightIdx !== null) {
          // close previous straight section
          let startOfStraight = track[startOfStraightIdx];
          let straightLength = utils.calculateSegment(startOfStraight, current);
          sections.push({
            type: 'straight',
            length: straightLength,
            points: [startOfStraight, current]
          });
          startOfStraightIdx = null;
        }
        const curv = utils.calculateCurve(current, next, nextNext);
        if (curv) {
          sections.push({
            type: 'curve',
            dir: curv.dir,
            radius: curv.radius,
            angle: curv.angle, // degrees
            points: [current, next, nextNext],
            center: curv.center
          });
          index++;
        } else {
          nullCurveCounter++;
        }
      }
      index++;
    }
    if (startOfStraightIdx !== null) {
      sections.push({
        type: 'straight',
        length: utils.calculateSegment(track[startOfStraightIdx], track[endOfStraightIdx]),
        points: [track[startOfStraightIdx], track[endOfStraightIdx]]
      });
    }
   
    this.fixTrackClosure();

    sections.forEach((s, idx) => { this.addSection(idx, s.type, s.length , s); });
    const finalTrackOutput = this.xmlTrackHeader + this.xml + this.closingXML;


    if (saveXMLalsoLocally) {
      try {
        fs.mkdirSync(OUTPUT_DIR_XML, { recursive: true });
        fs.writeFileSync(path.join(OUTPUT_DIR_XML, `output_${trackName}.xml`), finalTrackOutput, 'utf8');
        log.trace(`${this.logHeader} saved to ${path.join(OUTPUT_DIR_XML, `output_${trackName}.xml`)}`);
      } catch (err) {
        log.error(`${this.logHeader} Error creating directory or saving XML:`, err);
      }
    }
 
    return finalTrackOutput;
  }

  addSection(index, type, length, curv) {
    if (type === 'curve') {
      this.xml += `  <section name="c${index}">\n`;
      this.xml += `    <attstr name="type" val="${curv.dir}"/>\n`;
      this.xml += `    <attnum name="radius" unit="m" val="${curv.radius * this.trackScale}"/>\n`;
      this.xml += `    <attnum name="arc" unit="deg" val="${curv.angle}"/>\n`;
      this.xml += '  </section>\n';
    } else {
      this.xml += `  <section name="s${index}">\n`;
      this.xml += `    <attstr name="type" val="str"/>\n`;
      this.xml += `    <attnum name="lg" unit="m" val="${length * this.trackScale}"/>\n`;
      this.xml += '  </section>\n';
    }
  }

  /**
   * Function to adjust track sections to fix closure errors. Can be called multiple times.
   * @returns error of position and heading after correction
   */
  fixTrackClosure() {

    let maxFallBackIterations = 20;
    let initialPose = { x: 0, y: 0, heading: 0 };

    let initalTrackDelta = this.computeTrackDelta(initialPose);
    this.applyHeadingCorrection(initalTrackDelta.dtheta);

    let trackDelta = this.computeTrackDelta(initialPose);
    try {
      this.applyPositionCorrection(trackDelta, initialPose);
    } catch (e) {
      if (e instanceof PositionCorrectionError) {
        let i = 0;
        let delta = trackDelta;
        do {
          this.positionCorrectionFallback(delta, initialPose);
          delta = this.computeTrackDelta(initialPose);
          i++;
        } while ((delta.dx < -0.1 || delta.dx > 0.6 || Math.abs(delta.dy) > 0.1) && i < maxFallBackIterations);
        log.debug(`${this.logHeader} Cannot apply position correction: ${e.message} Applying fallback method. Iterations: ${i}`);
      } else {
        throw e;
      }
    }
    // recalculate final pose after position correction
    let finalTrackDelta = this.computeTrackDelta(initialPose);
    // log both errors
    if (finalTrackDelta.dx < -0.1 || finalTrackDelta.dx > 0.6 || Math.abs(finalTrackDelta.dy) > 0.1) {
      // dx and dy will have opposite sign in the trackgen output
      log.warn(`${this.logHeader} Track position closure error too high after correction: dx=${finalTrackDelta.dx.toFixed(4)}, dy=${finalTrackDelta.dy.toFixed(4)}, trackLength=${this.track.length}`);
    }

    return finalTrackDelta;
  }

  calculateInitialPose(sections = this.sections) {
    let heading;
    if (sections[0].type === 'straight') {
      heading = utils.normalizeVector({ x: sections[0].points[1].x - sections[0].points[0].x, y: sections[0].points[1].y - sections[0].points[0].y });
    }
    else {
      heading = utils.calculateCurveInitialHeading(
        sections[0].points[0],
        sections[0].points[2],
        sections[0].radius,
        sections[0].dir
      );
    }

    // return heading converted to radians
    return { x: 0, y: 0, heading: Math.atan2(heading.y, heading.x) };
  }

  /**
   * Distributes heading error correction across all curve sections.
   * @param {*} dtheta 
   * @param {*} sections 
   * @returns 
   */
  applyHeadingCorrection(dtheta, sections = this.sections) {
    const curves = sections.filter(s => s.type === 'curve');
    if (curves.length === 0) return; // nothing to fix

    // compute total absolute angle for weighting
    const totalAbsAngle = curves.reduce((a, c) => a + Math.abs(c.angle), 0);
    for (const c of curves) {
      const weight = Math.abs(c.angle) / totalAbsAngle;
      const deltaDegree = dtheta * (180 / Math.PI);
      const dirSign = c.dir === 'lft' ? 1 : -1;
      const deltaAngle = -deltaDegree * weight * dirSign;
      c.angle += deltaAngle;
    }
  }


  applyPositionCorrection(error, startPose = { x: 0, y: 0, heading: 0 }, sections = this.sections) {
    const straights = [];
    let pose = { ...startPose };

    // Collect all straights with their starting heading (same as your original function)
    for (const s of sections) {
      if (s.type === 'straight') {
        straights.push({ section: s, heading: pose.heading });
        pose.x += s.length * Math.cos(pose.heading);
        pose.y += s.length * Math.sin(pose.heading);
      } else if (s.type === 'curve') {
        const dirSign = s.dir === 'lft' ? 1 : -1;
        const angRad = (s.angle * Math.PI) / 180;
        const cx = pose.x - dirSign * s.radius * Math.sin(pose.heading);
        const cy = pose.y + dirSign * s.radius * Math.cos(pose.heading);
        pose.heading += dirSign * angRad;
        pose.x = cx + dirSign * s.radius * Math.sin(pose.heading);
        pose.y = cy - dirSign * s.radius * Math.cos(pose.heading);
      }
    }

    if (straights.length === 0) return; // nothing to fix

    const ex = -error.dx; // We want to move the endpoint back by the error amount
    const ey = -error.dy;

    // --- Start of new logic: Least-Squares Adjustment ---

    // We are solving the system A * deltaL = E for the vector of length changes deltaL.
    // Where A = [[cos(h1), cos(h2), ...], [sin(h1), sin(h2), ...]]
    // and E = [ex, ey].
    // The solution that minimizes the sum of squares of deltaL is:
    // deltaL = A_transpose * inverse(A * A_transpose) * E


    // --- Start of new and modified logic ---

    // Handle the edge case of a single straight section
    if (straights.length === 1) {
      throw new PositionCorrectionError("Only one straight section present.");
    }

    // 1. Calculate the components of the 2x2 matrix C = A * A_transpose
    let s_c2 = 0, s_s2 = 0, s_sc = 0;
    for (const { heading } of straights) {
      const c = Math.cos(heading);
      const s = Math.sin(heading);
      s_c2 += c * c;
      s_s2 += s * s;
      s_sc += s * c;
    }

    // 2. Calculate the determinant of C. If it's near zero, the system can't be solved
    // (this happens if all straights are parallel).
    const det = s_c2 * s_s2 - s_sc * s_sc;
    if (Math.abs(det) < 1e-9) {
      throw new PositionCorrectionError("All straight sections are parallel.");
    }
    const inv_det = 1 / det;

    // 3. Calculate the vector V = inverse(C) * E
    const v0 = inv_det * (ex * s_s2 - ey * s_sc);
    const v1 = inv_det * (-ex * s_sc + ey * s_c2);

    // 4. Calculate the change in length for each straight: deltaL_i = D_i * V
    for (const { section, heading } of straights) {
      const c = Math.cos(heading);
      const s = Math.sin(heading);
      const deltaLen = c * v0 + s * v1;
      section.length += deltaLen;
    }
    // --- End of new logic ---
  }


  positionCorrectionFallback(error, startPose = { x: 0, y: 0, heading: 0 }, sections = this.sections) {
    const straights = [];
    let pose = { ...startPose };

    // collect all straights with their heading
    for (const s of sections) {
      if (s.type === 'straight') {
        straights.push({ section: s, heading: pose.heading });
        pose.x += s.length * Math.cos(pose.heading);
        pose.y += s.length * Math.sin(pose.heading);
      } else if (s.type === 'curve') {
        const dirSign = s.dir === 'lft' ? 1 : -1;
        const angRad = (s.angle * Math.PI) / 180;
        const cx = pose.x - dirSign * s.radius * Math.sin(pose.heading);
        const cy = pose.y + dirSign * s.radius * Math.cos(pose.heading);
        pose.heading += dirSign * angRad;
        pose.x = cx + dirSign * s.radius * Math.sin(pose.heading);
        pose.y = cy - dirSign * s.radius * Math.cos(pose.heading);
      }
    }

    if (straights.length === 0) return; // nothing to fix

    // convert closure error to vector
    const ex = -error.dx; // we want to move back by -dx
    const ey = -error.dy;

    const totalProjectedLength = straights.reduce((a, s) => {
      const dirx = Math.cos(s.heading);
      const diry = Math.sin(s.heading);
      const proj = ex * dirx + ey * diry;
      return a + Math.abs(proj) * s.section.length;
    }, 0);

    // distribute correction
    for (const { section, heading } of straights) {

      // how much of the error projects along this straight's direction
      const dirx = Math.cos(heading);
      const diry = Math.sin(heading);
      const proj = ex * dirx + ey * diry; // component along this straight

      const w = (Math.abs(proj) * section.length) / totalProjectedLength;

      // apply proportional correction (damped)
      const deltaLen = proj * w;


      section.length += deltaLen;
    }
  }

  /**
   * Calculates the final pose after following the track sections from a given start pose.
   * @param {*} startPose 
   * @param {*} sections 
   * @returns 
   */
  calculateFinalPose(startPose = { x: 0, y: 0, heading: 0 }, sections = this.sections) {
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
    pose.heading = Math.atan2(Math.sin(pose.heading), Math.cos(pose.heading));

    return pose;
  }

  /**
   * Computes the track delta (dx, dy, dtheta) after following the sections from a given start pose.
   * @param {*} startPose 
   * @param {*} sections 
   * @returns
   */
  computeTrackDelta(startPose = { x: 0, y: 0, heading: 0 }, sections = this.sections) {
    const finalPose = this.calculateFinalPose(startPose);
    return {
      dx: finalPose.x,
      dy: finalPose.y,
      dtheta: finalPose.heading - startPose.heading,
    };
  }

  getLength() {
    return this.sections.reduce((acc, s) => acc + (s.type === 'straight' ? s.length : (s.radius * (s.angle * Math.PI / 180))), 0) * this.trackScale;
  }

}

export default TorcsXMLGenerator;


