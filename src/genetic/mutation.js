import { prng_alea } from '../lib/esm-seedrandom/alea.min.mjs';

export function mutationVoronoi(individual, intensity, seed = null) {
  let random = Math.random;
  if (seed !== null) random = prng_alea(seed);

  const selectedCells = individual.selectedCells.map(cell => ({ ...cell.site }));
  const dataSet = [...individual.dataSet];
  const randomIndex = Math.floor(random() * selectedCells.length);
  const deltaX = intensity * (random() - 0.5) * 2;
  const deltaY = intensity * (random() - 0.5) * 2;
  const dataSetIndex = dataSet.findIndex(point =>
    point.x === selectedCells[randomIndex].x &&
    point.y === selectedCells[randomIndex].y
  ); 
  if (dataSetIndex === -1) {
    throw new Error('Selected cell not found in dataset');
  }

  selectedCells[randomIndex].x += deltaX;
  selectedCells[randomIndex].y += deltaY;

  // change the point inplace
  dataSet[dataSetIndex] = selectedCells[randomIndex];

  return { ds: dataSet, sel: selectedCells };
}

//let's move randomly a point in the convexHull
export function mutationConvexHull(individual, intensity, seed = null) {
  let random = Math.random;
  if (seed !== null) random = prng_alea(seed);
  const dataSetHull = [...individual.dataSetHull];
  const dataSet = [...individual.dataSet];

  const randomIndex = Math.floor(random() * dataSetHull.length);
  const originalPoint = { ...dataSetHull[randomIndex] };

  dataSetHull[randomIndex].x += intensity * (random() - 0.5) * 2;
  dataSetHull[randomIndex].y += intensity * (random() - 0.5) * 2;

  const dataSetIndex = dataSet.findIndex(point =>
    point.x === originalPoint.x && point.y === originalPoint.y
  );

  if (dataSetIndex !== -1) {
    dataSet[dataSetIndex] = dataSetHull[randomIndex];
  }

  return { ds: dataSet };
}
