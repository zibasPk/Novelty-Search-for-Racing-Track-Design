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
    // Safety net: clamp any remaining negative-length straights to 0.001
    // TORCS silently skips segments with negative length (step count becomes <= 0
    // in CreateSegRing), causing a mismatch between JS-computed and TORCS-computed deltas.
    for (const s of this.sections) {
      if (s.type === 'straight' && s.length <= 0) {
        log.debug(`${this.logHeader} Clamping negative straight length ${s.length.toFixed(4)} to 0`);
        s.length = 0.001;
      }
    }

    // recalculate final pose after position correction (and clamping)
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

    // Collect all straights with their starting heading
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

    if (straights.length === 0) return;

    const ex = -error.dx;
    const ey = -error.dy;

    // Least-Squares with non-negative length constraints (active-set method).
    // TORCS skips straight segments with negative length (step count becomes <= 0),
    // so we must ensure all straight lengths remain >= 0 after correction.

    if (straights.length === 1) {
      throw new PositionCorrectionError("Only one straight section present.");
    }

    // Active set: iteratively pin straights that would go negative to length 0,
    // then re-solve for the remaining straights.
    let activeStraights = straights.map((s, i) => ({ ...s, idx: i }));
    const pinnedDeltas = new Map(); // index -> deltaLen applied (= -originalLength)

    for (let iter = 0; iter <= straights.length; iter++) {
      if (activeStraights.length === 0) break;

      // Compute remaining error after accounting for pinned straights
      let rex = ex, rey = ey;
      for (const [idx, dl] of pinnedDeltas) {
        rex -= dl * Math.cos(straights[idx].heading);
        rey -= dl * Math.sin(straights[idx].heading);
      }

      if (activeStraights.length === 1) {
        // Single active straight: project error onto its direction
        const { section, heading } = activeStraights[0];
        const c = Math.cos(heading);
        const s = Math.sin(heading);
        const deltaLen = (rex * c + rey * s) / (c * c + s * s);
        if (section.length + deltaLen < 0) {
          section.length = 0;
        } else {
          section.length += deltaLen;
        }
        break;
      }

      // Least-squares for multiple active straights
      let s_c2 = 0, s_s2 = 0, s_sc = 0;
      for (const { heading } of activeStraights) {
        const c = Math.cos(heading);
        const s = Math.sin(heading);
        s_c2 += c * c;
        s_s2 += s * s;
        s_sc += s * c;
      }

      const det = s_c2 * s_s2 - s_sc * s_sc;
      if (Math.abs(det) < 1e-9) {
        throw new PositionCorrectionError("All straight sections are parallel.");
      }
      const inv_det = 1 / det;
      const v0 = inv_det * (rex * s_s2 - rey * s_sc);
      const v1 = inv_det * (-rex * s_sc + rey * s_c2);

      // Check if any active straight would go negative
      let worstIdx = -1;
      let worstNewLen = Infinity;
      const deltas = [];
      for (let i = 0; i < activeStraights.length; i++) {
        const { section, heading } = activeStraights[i];
        const c = Math.cos(heading);
        const s = Math.sin(heading);
        const deltaLen = c * v0 + s * v1;
        deltas.push(deltaLen);
        const newLen = section.length + deltaLen;
        if (newLen < 0 && newLen < worstNewLen) {
          worstNewLen = newLen;
          worstIdx = i;
        }
      }

      if (worstIdx === -1) {
        // All lengths remain non-negative: apply corrections
        for (let i = 0; i < activeStraights.length; i++) {
          activeStraights[i].section.length += deltas[i];
        }
        break;
      } else {
        // Pin the most violated straight to 0 and re-solve
        const pinned = activeStraights[worstIdx];
        const pinnedDelta = -pinned.section.length; // set to 0
        pinnedDeltas.set(pinned.idx, pinnedDelta);
        pinned.section.length = 0;
        activeStraights.splice(worstIdx, 1);
        log.debug(`${this.logHeader} Pinned straight at index ${pinned.idx} to 0 (would have gone to ${worstNewLen.toFixed(2)})`);
      }
    }
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

    // distribute correction (clamped to prevent negative lengths)
    for (const { section, heading } of straights) {

      // how much of the error projects along this straight's direction
      const dirx = Math.cos(heading);
      const diry = Math.sin(heading);
      const proj = ex * dirx + ey * diry; // component along this straight

      const w = (Math.abs(proj) * section.length) / totalProjectedLength;

      // apply proportional correction (damped), clamped to prevent negative lengths
      const deltaLen = Math.max(proj * w, -section.length);

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


