import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import * as utils from './utils.js';
import { OUTPUT_DIR_XML } from './constants.js';
import { log, trace } from 'console';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const XML_TRACK_HEADER = fs.readFileSync(path.join(__dirname, 'startTrackTemplate.xml'), 'utf8');
const CLOSING_XML = "</section>\n</section>\n</params>";
let xml = '';

//return XML data ready for trackGen parsing  
// saveXMLalsoLocally is used for testing, it prints at local level the XML as "output.xml"
export function exportTrackToXML(track, startIndex = 0, saveXMLalsoLocally = false, trackName = 'default') {
  xml = '';
  const threshold = 0.01;
  const sections = [];

  let startOfStraightIdx = null;
  let endOfStraightIdx = null;

  for (let index = startIndex; index < startIndex + track.length - 2; index++) {
    const i = (index) % track.length;
    const i_next = (index + 1) % track.length;
    const i_nextnext = (index + 2) % track.length;

    const current = track[i];
    const next = track[i_next];
    const nextNext = track[i_nextnext];

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
      }
    }
  }

  if (startOfStraightIdx !== null) {
    sections.push({
      type: 'straight',
      length: utils.calculateSegment(track[startOfStraightIdx], track[endOfStraightIdx]),
      points: [track[startOfStraightIdx], track[endOfStraightIdx]]
    });
  }
  
  let error;
  let i = 0;
  do {
    error = utils.fixTrackClosure(sections);
    i++;
    // log(`Closure correction iteration ${i}: dx=${error.dx.toFixed(4)}, dy=${error.dy.toFixed(4)}, dtheta=${error.dtheta.toFixed(6)}`);
  } while (Math.abs(error.dx) > 0.15 || Math.abs(error.dy) > 0.15 || Math.abs(error.dtheta) > 0.005);
  log(`Final closure error after ${i} iterations: dx=${error.dx.toFixed(4)}, dy=${error.dy.toFixed(4)}, dtheta=${error.dtheta.toFixed(6)}`);


  sections.forEach((s, idx) => { addSection(idx, s.type, s.length, s); });
  const finalTrackOutput = XML_TRACK_HEADER + xml + CLOSING_XML;

  if (saveXMLalsoLocally) {
    try {
      fs.mkdirSync(OUTPUT_DIR_XML, { recursive: true });
      fs.writeFileSync(path.join(OUTPUT_DIR_XML, `output_${trackName}.xml`), finalTrackOutput, 'utf8');
    } catch (err) {
      console.error('Error creating directory or saving XML:', err);
    }
  }

  return finalTrackOutput;
}

function addSection(index, type, length, curv) {
  if (type === 'curve') {
    xml += `  <section name="c${index}">\n`;
    xml += `    <attstr name="type" val="${curv.dir}"/>\n`;
    xml += `    <attnum name="radius" unit="m" val="${curv.radius}"/>\n`;
    xml += `    <attnum name="arc" unit="deg" val="${curv.angle}"/>\n`;
    xml += '  </section>\n';
  } else {
    xml += `  <section name="s${index}">\n`;
    xml += `    <attstr name="type" val="str"/>\n`;
    xml += `    <attnum name="lg" unit="m" val="${length}"/>\n`;
    xml += '  </section>\n';
  }
}