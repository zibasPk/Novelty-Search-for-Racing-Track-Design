import {
  BBOX,
} from '../constants.js';
import { generateTrack } from '../../trackGen/trackGenerator.js';
import { TorcsXMLGenerator } from '../../trackGen/torcsXMLGenerator.js';
import { simulate } from '../../sim/simulateTrack.js';
import log from "loglevel"
import fs from 'fs/promises';


// let startSeed = 0;
// const trackLengths = [];

// for (let seed = startSeed; seed < startSeed + 2000; seed++) {
//   log.setLevel("warn");
//   let result = await generateTrack(
//     "voronoi", BBOX, seed, (seed % 8) + 1,
//     true, [], []
//   );
//   trackLengths.push(result.track.length);
//   log.setLevel("debug");
//   try {
//     // const trackXml = xml.exportTrackToXML(result.track, 0, true, seed);
//     let xmlGenerator = new TorcsXMLGenerator(result.track, seed);
//     xmlGenerator.generateXML(0, true);
//   } catch (e) {
//     log.error(`Error generating xml for track seed ${seed} : ${e.message}`);
//   }
// }

// for all files in the data / voronoi / fitted folder generate the xml file
// import fs from 'fs/promises';
// const fittedDir = 'data/voronoi/fitted/';
// const files = await fs.readdir(fittedDir);
// for (const file of files) {
//   if (file.endsWith('.json')) {
//     const filePath = fittedDir + file;
//     const data = await fs.readFile(filePath, 'utf-8');
//     const trackData = JSON.parse(data);

//     log.setLevel("debug");
//     try {
//       await simulate(
//         trackData.mode,
//         trackData.selectedCells.length,
//         trackData.dataSet,
//         trackData.selectedCells,
//         trackData.id,
//         false
//       );
//     } catch (e) {
//       log.error(`Error simulating track from file ${file} : ${e.message}`);
//     }
//   }
// }


// load from data/voronoi/json the file named 123.json and generate the xml file
const jsonDir = 'data/voronoi/json/';
const filename = '109.59621462502487.json';
const filePath = jsonDir + filename;

const data = await fs.readFile(filePath, 'utf-8');
const trackData = JSON.parse(data);

log.setLevel("debug");


try {
  // generate track json
  const trackResults = await generateTrack(
    trackData.mode, BBOX, trackData.id, trackData.selectedCells.length,
    false, trackData.dataSet, trackData.selectedCells
  );

  const seed = trackData.id;
  // translate to XML for TORCS
  const xmlGenerator = new TorcsXMLGenerator(trackResults.track, seed);
  const trackXml = xmlGenerator.generateXML(0, true);

} catch (e) {
  log.error(`Error simulating track from file : ${e.message}`);
}
