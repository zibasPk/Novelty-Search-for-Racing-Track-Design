import {
  BBOX,
  RngMode,
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
//   let result = await generateTrack({
//     mode: "voronoi", bbox: BBOX, seed, trackSize: (seed % 8) + 1,
//     saveJSON: true, dataSet: [], selected: []
//   });
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
// const fittedDir = 'data/voronoi/fitted/';
// const files = await fs.readdir(fittedDir);
// log.setLevel("info");

// for (const file of files) {
//   if (file.endsWith('.json')) {
//     const filePath = fittedDir + file;
//     const data = await fs.readFile(filePath, 'utf-8');
//     const trackData = JSON.parse(data);

//     genJsonAndXml(trackData);
//   }
// }


// load from data/voronoi/json the file named 123.json and generate the xml file
const jsonDir = 'data/voronoi/fitted/';
const filename = '12850.json';
const filePath = jsonDir + filename;

const data = await fs.readFile(filePath, 'utf-8');
const trackData = JSON.parse(data);

log.setLevel("debug");

genJsonAndXml(trackData);


async function genJsonAndXml(trackData) {
  try {
    // generate track json
    const trackResults = await generateTrack({
      mode: trackData.mode, bbox: BBOX, seed: trackData.id, trackSize: trackData.trackSize,
      saveJSON: false, dataSet: trackData.dataSet, selected: trackData.selectedCells, rngMode: RngMode.PERLIN
    });
    const seed = trackData.id;
    // translate to XML for TORCS
    const xmlGenerator = new TorcsXMLGenerator(trackResults.track, seed);

  } catch (e) {
    log.error(`Error generating track from file : ${e.message}`);
  }
}