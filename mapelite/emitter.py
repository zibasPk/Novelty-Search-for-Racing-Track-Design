# emitter.py

import numpy as np
import requests
import random

from ribs.emitters import EmitterBase

from config import BASE_URL, BATCH_SIZE, INIT_POPULATION, SOLUTION_DIM, INVALID_SCORE, GENERATION_MODE, TRACK_SIZE_RANGE
import utils

class CustomEmitter(EmitterBase):
    """
    A custom emitter for MAP-Elites that handles initial generation, 
    API-based mutation, and API-based crossover.
    """
    def __init__(self, archive, solution_dim, batch_size=BATCH_SIZE, bounds=None):
        super().__init__(archive, solution_dim=solution_dim, bounds=bounds)
        self.batch_size = batch_size
        self.iteration = 0

    def ask(self):
        """Generates a batch of solutions for the scheduler to evaluate."""
        self.iteration += 1
        print(f"Emitter.ask() called for iteration {self.iteration}")
        
        if self.iteration <= INIT_POPULATION:
            # Initial random generation
            out = []
            # Note: We generate one at a time and append until batch_size is met
            # The scheduler's main loop will call ask() until it has enough solutions (INIT_POPULATION)
            # and then call ask() with BATCH_SIZE for subsequent iterations.
            
            # Here we fill up one dask batch_size if it's <= INIT_POPULATION
            for _ in range(self.batch_size):
                sol = self.generate_solution(self.iteration - 1)
                arr = utils.solution_to_array(sol)
                if arr is not None:
                    out.append(arr)
                else:
                    out.append(np.full(SOLUTION_DIM, INVALID_SCORE))
            return np.array(out)
        else:
            # Main QD loop: 50/50 mutation or crossover
            if random.random() < 0.5:
                # Mutate (returns BATCH_SIZE solutions)
                return self.mutate_solutions()
            else:
                # Crossover (returns BATCH_SIZE solutions, created from BATCH_SIZE // 2 pairs)
                return self.crossover_solutions()
    
    def generate_solution(self, iteration):
        """Generates a new track solution by calling the external API."""
        # print(f"Generating solution for iteration {iteration}") # Mute: too chatty
        try:
            response = requests.post(
                f"{BASE_URL}/generate",
                json={
                    "id": iteration + random.random(),
                    "mode": GENERATION_MODE,
                    "trackSize": random.randint(TRACK_SIZE_RANGE[0], TRACK_SIZE_RANGE[1])
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error generating solution for iteration {iteration}: {e}")
            return None 
    
    def mutate_solutions(self):
        """Mutates existing elite solutions via the external API."""
        print(f"Mutating solutions for iteration {self.iteration}")
        
        # Sample BATCH_SIZE parents from the archive
        parents = self.archive.sample_elites(self.batch_size)
        out = []
        
        for i in range(self.batch_size):
            arr = parents["solution"][i]
            sol = utils.array_to_solution(arr)
            
            try:
                response = requests.post(
                    f"{BASE_URL}/mutate",
                    json={
                        "individual": sol,
                        "intensityMutation": 20 # Constant mutation intensity
                    },
                    timeout=60
                )
                response.raise_for_status()
                
                mutated = response.json().get("mutated", {})
                
                # Assign a unique, iteration-based ID for tracking
                frac = utils.get_fractional_part(sol["id"])
                mutated["id"] = self.iteration - 1 + frac
                
                mutated_arr = utils.solution_to_array(mutated)
                
                if mutated_arr is not None:
                    out.append(mutated_arr)
                    # print(f"Mutated ID={sol['id']} to ID={mutated['id']}") # Mute: too chatty
                else:
                    out.append(np.full(SOLUTION_DIM, INVALID_SCORE))
            
            except requests.RequestException as e:
                print(f"Error mutating solution ID={sol['id']}: {e}")
                out.append(np.full(SOLUTION_DIM, INVALID_SCORE))
        
        return np.array(out)

    def crossover_solutions(self):
        """Performs crossover between sampled elite solutions via the external API."""
        print(f"Crossover solutions for iteration {self.iteration}")
        out = []
        
        # Generate BATCH_SIZE solutions from BATCH_SIZE // 2 pairs
        for _ in range(self.batch_size // 2):
            try:
                # 1. Sample two distinct parents
                while True:
                    parents = self.archive.sample_elites(2)
                    sol1 = utils.array_to_solution(parents["solution"][0])
                    sol2 = utils.array_to_solution(parents["solution"][1])
                    if sol1["id"] != sol2["id"]:
                        break
                        
                # 2. Call the external crossover API
                response = requests.post(
                    f"{BASE_URL}/crossover",
                    json={
                        "mode": GENERATION_MODE,
                        "parent1": sol1,
                        "parent2": sol2
                    },
                    timeout=60
                )
                response.raise_for_status()
                
                offspring = response.json().get("offspring", {})
                
                # 3. Create two new children (assuming crossover produces 2, though the API returns 1 'offspring' as one merged result)
                # Note: The original logic only extracts one 'offspring' dict, let's stick to generating one solution per loop iteration (total BATCH_SIZE // 2 iterations)
                
                f1 = utils.get_fractional_part(sol1["id"])
                f2 = utils.get_fractional_part(sol2["id"])
                frac = (f1 + f2) % 1
                child_id = self.iteration - 1 + frac
                
                child_sol = {
                    "id": child_id,
                    "mode": GENERATION_MODE,
                    "trackSize": len(offspring.get("sel", [])),
                    "dataSet": offspring.get("ds", []),
                    "selectedCells": offspring.get("sel", [])
                }
                
                child_arr = utils.solution_to_array(child_sol)
                
                if child_arr is not None:
                    out.append(child_arr)
                    # print(f"Crossover Parent1 ID={sol1['id']}, Parent2 ID={sol2['id']} => Child ID={child_id}") # Mute: too chatty
                else:
                    out.append(np.full(SOLUTION_DIM, INVALID_SCORE))
            
            except requests.RequestException as e:
                print(f"Error during crossover: {e}")
                out.append(np.full(SOLUTION_DIM, INVALID_SCORE))
        
        if len(out) < self.batch_size:
            if len(out) == self.batch_size // 2 and self.batch_size % 2 == 0:
                 out = out + out 
                 out = out[:self.batch_size] # just in case

        return np.array(out)