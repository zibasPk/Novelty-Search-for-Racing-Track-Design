# emitter.py

import numpy as np
import requests
import random
from qd.logging_config import get_logger

from ribs.emitters import EmitterBase

from qd.config import BASE_URL, BATCH_SIZE, DEFAULT_START_ITER, RANDOM_POPULATION_ITERS, GENERATION_MODE, TRACK_SIZE_RANGE, RngMode
import qd.utils as utils

log = get_logger("emitter")

class CustomEmitter(EmitterBase):
    """
    A custom emitter for MAP-Elites that handles initial generation, 
    API-based mutation, and API-based crossover.
    """
    def __init__(self, archive, solution_dim, batch_size=BATCH_SIZE, bounds=None, seed=None):
        super().__init__(archive, solution_dim=solution_dim, bounds=bounds)
        self.batch_size = batch_size
        self.iteration = DEFAULT_START_ITER
        self.seed = seed
        self._rng = random.Random(seed)

    def ask(self):
        """Generates a batch of solutions for the scheduler to evaluate."""
        log.info("Emitter.ask called", iteration=self.iteration)
        try:
            if self.iteration < RANDOM_POPULATION_ITERS:
                # Initial random generation
                out = []  
                for _ in range(self.batch_size):
                    sol, id = self.generate_solution()
                    arr = utils.solution_to_array(sol)
                    if arr is not None:
                        out.append(arr)
                    else:
                        out.append(utils.invalid_solution_array(id))
                return np.array(out)
            else:
                if self.iteration % 2 == 0:
                    # Mutate (returns BATCH_SIZE solutions)
                    return self.mutate_solutions()
                else:
                    # Crossover (returns BATCH_SIZE solutions, created from BATCH_SIZE // 2 pairs)
                    return self.crossover_solutions()
        finally:
            self.iteration += 1
        
    def generate_solution(self):
        """Generates a new track solution by calling the external API."""
        rngMode = RngMode.UNIFORM if self.iteration % 2 == 0 else RngMode.PERLIN
        id = self.iteration + self._rng.random()  # Unique ID for tracking, based on iteration and randomness
        
        try:
            response = requests.post(
                f"{BASE_URL}/generate",
                json={
                    "id": id,
                    "mode": GENERATION_MODE,
                    "trackSize": self._rng.randint(TRACK_SIZE_RANGE[0], TRACK_SIZE_RANGE[1]),
                    "rngMode": rngMode
                },
                timeout=60
            )
            if not response.ok:
                    raise Exception(f"API error {response.status_code}: {response.text}")
                
            sol = response.json()
            sol["rngMode"] = rngMode  # persist rngMode in the genome
            return sol, id
        except Exception as e:
            log.warning("Error generating solution", iteration=self.iteration, error=str(e))
            return None, id
    
    def mutate_solutions(self):
        """Mutates existing elite solutions via the external API."""
        log.debug("Mutating solutions", iteration=self.iteration)
        
        
        # Sample BATCH_SIZE parents from the archive
        parents = self.archive.sample_elites(self.batch_size)
        out = []
        
        for i in range(self.batch_size):
            arr = parents["solution"][i]
            sol = utils.array_to_solution(arr)
            seed = self.iteration + self._rng.random()  # Unique seed for mutation based on iteration and randomness
            
            try:
                response = requests.post(
                    f"{BASE_URL}/mutate",
                    json={
                        "individual": sol,
                        "intensityMutation": 20, # Constant mutation intensity
                        "genetic_seed": seed
                    },
                    timeout=60
                )
                if not response.ok:
                    raise Exception(f"API error {response.status_code}: {response.text}")
                
                mutated = response.json().get("mutated", {})
                
                # Assign a unique, iteration-based ID for tracking
                mutated["id"] = seed
                mutated["rngMode"] = sol.get("rngMode", RngMode.UNIFORM)  # inherit rngMode from parent
                
                mutated_arr = utils.solution_to_array(mutated)
                
                if mutated_arr is not None:
                    out.append(mutated_arr)
                    # print(f"Mutated ID={sol['id']} to ID={mutated['id']}") # Mute: too chatty
                else:
                    out.append(utils.invalid_solution_array(seed))
            
            except Exception as e:
                log.warning("Error mutating solution", sol_id=sol["id"], error=str(e))
                out.append(utils.invalid_solution_array(seed))
        
        return np.array(out)

    def crossover_solutions(self):
        """Performs crossover between sampled elite solutions via the external API."""
        log.debug("Crossover solutions", iteration=self.iteration)
        
        if self.archive.stats.num_elites < 2:
            log.warning("Not enough elites for crossover", num_elites=self.archive.stats.num_elites)
            return np.array([utils.invalid_solution_array(-1) for _ in range(self.batch_size)])
        
        out = []
        
        # Generate BATCH_SIZE solutions
        for _ in range(self.batch_size):
            try:         
                seed = self.iteration + self._rng.random()
                # Sample two distinct parents
                i = 0
                while True:
                    parents = self.archive.sample_elites(2)
                    sol1 = utils.array_to_solution(parents["solution"][0])
                    sol2 = utils.array_to_solution(parents["solution"][1])
                    if sol1["id"] != sol2["id"]:
                        break
                    i += 1
                    if i > 1000:  # Avoid infinite loop if archive is small
                        raise Exception("Unable to sample two distinct parents for crossover")

                response = requests.post(
                    f"{BASE_URL}/crossover",
                    json={
                        "mode": GENERATION_MODE,
                        "parent1": sol1,
                        "parent2": sol2,
                        "genetic_seed": seed
                    },
                    timeout=60
                )
                if not response.ok:
                    raise Exception(f"API error {response.status_code}: {response.text}")
                
                offspring = response.json().get("offspring", {})
                
                child_id = seed
               
                
                child_sol = {
                    "id": child_id,
                    "mode": GENERATION_MODE,
                    "trackSize": len(offspring.get("sel", [])),
                    "dataSet": offspring.get("ds", []),
                    "selectedCells": offspring.get("sel", []),
                    "rngMode": sol1.get("rngMode", RngMode.UNIFORM)  # inherit rngMode from parent1
                }
                
                child_arr = utils.solution_to_array(child_sol)
                
                if child_arr is not None:
                    out.append(child_arr)
                    # print(f"Crossover Parent1 ID={sol1['id']}, Parent2 ID={sol2['id']} => Child ID={child_id}") # Mute: too chatty
                else:
                    raise Exception("Invalid child solution format received from crossover API")
            
            except Exception as e:
                log.error("Error during crossover", error=str(e))
                out.append(utils.invalid_solution_array(seed))

        return np.array(out)