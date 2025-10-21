export class SimulationTimeoutError extends Error {
  constructor(message = 'Simulation timeout') {
    super(message);
    this.name = 'SimulationTimeoutError';
  }
}