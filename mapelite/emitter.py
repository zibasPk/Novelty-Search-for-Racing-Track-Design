# emitter.py

import numpy as np
import requests
import random

from ribs.emitters import EmitterBase

from config import BASE_URL, BATCH_SIZE, RANDOM_POPULATION_ITERS, SOLUTION_DIM, INVALID_SCORE, GENERATION_MODE, TRACK_SIZE_RANGE, RngMode
import utils

class CustomEmitter(EmitterBase):
    """
    A custom emitter for MAP-Elites that handles initial generation, 
    API-based mutation, and API-based crossover.
    """
    def __init__(self, archive, solution_dim, batch_size=BATCH_SIZE, bounds=None, seed=None):
        super().__init__(archive, solution_dim=solution_dim, bounds=bounds)
        self.batch_size = batch_size
        self.iteration = 0
        self.seed = seed
        # Use a dedicated RNG instance seeded deterministically so the emitter's
        # randomness is self-contained and not affected by global random state.
        self._rng = random.Random(seed)

    def ask(self):
        """Generates a batch of solutions for the scheduler to evaluate."""
        self.iteration += 1
        print(f"Emitter.ask() called for iteration {self.iteration}")
        
        if self.iteration <= RANDOM_POPULATION_ITERS:
            # Initial random generation
            out = []
            # Note: We generate one at a time and append until batch_size is met
            # The scheduler's main loop will call ask() until it has enough solutions (INIT_POPULATION)
            # and then call ask() with BATCH_SIZE for subsequent iterations.
            
            # Here we fill up one dask batch_size if it's <= INIT_POPULATION
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
    
    def generate_solution(self):
        """Generates a new track solution by calling the external API."""
        rngMode = RngMode.UNIFORM if self.iteration % 2 == 0 else RngMode.PERLIN
        id = self.iteration - 1 + self._rng.random()  # Unique ID for tracking, based on iteration and randomness
        
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
            print(f"Error generating solution for iteration {self.iteration}: {e}")
            return None, id
    
    def mutate_solutions(self):
        """Mutates existing elite solutions via the external API."""
        print(f"Mutating solutions for iteration {self.iteration}")
        
        
        # Sample BATCH_SIZE parents from the archive
        parents = self.archive.sample_elites(self.batch_size)
        out = []
        
        for i in range(self.batch_size):
            arr = parents["solution"][i]
            sol = utils.array_to_solution(arr)
            seed = self.iteration - 1 + self._rng.random()  # Unique seed for mutation based on iteration and randomness
            
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
                frac = utils.get_fractional_part(sol["id"])
                mutated["id"] = seed
                mutated["rngMode"] = sol.get("rngMode", RngMode.UNIFORM)  # inherit rngMode from parent
                
                mutated_arr = utils.solution_to_array(mutated)
                
                if mutated_arr is not None:
                    out.append(mutated_arr)
                    # print(f"Mutated ID={sol['id']} to ID={mutated['id']}") # Mute: too chatty
                else:
                    out.append(utils.invalid_solution_array(seed))
            
            except Exception as e:
                print(f"Error mutating solution ID={sol['id']}: {e}")
                out.append(utils.invalid_solution_array(seed))
        
        return np.array(out)

    def crossover_solutions(self):
        """Performs crossover between sampled elite solutions via the external API."""
        print(f"Crossover solutions for iteration {self.iteration}")
        
        if self.archive.stats.num_elites < 2:
            # Not enough elites to perform crossover, return invalid solutions
            print("Not enough elites for crossover, returning invalid solutions")
            return np.array([utils.invalid_solution_array() for _ in range(self.batch_size)])
        
        out = []
        
        # Generate BATCH_SIZE solutions from BATCH_SIZE
        for _ in range(self.batch_size):
            try:
                # 1. Sample two distinct parents
                while True:
                    parents = self.archive.sample_elites(2)
                    sol1 = utils.array_to_solution(parents["solution"][0])
                    sol2 = utils.array_to_solution(parents["solution"][1])
                    if sol1["id"] != sol2["id"]:
                        break
                    
                # Assign a unique, iteration-based ID for tracking
                seed = self.iteration - 1 + self._rng.random()
                   
                # 2. Call the external crossover API
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
                
                child_id = seed  # Unique ID for the child based on iteration and randomness
               
                
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
                print(f"Error during crossover: {e}")
                out.append(utils.invalid_solution_array(seed))

        return np.array(out)