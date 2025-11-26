export class SimulationTimeoutError extends Error {
  constructor(message = 'Simulation timeout') {
    super(message);
    this.name = 'SimulationTimeoutError';
  }
}

export class PositionCorrectionError extends Error {
  constructor(message = 'Position correction failed') {
    super(message);
    this.name = 'PositionCorrectionError';
  }
}