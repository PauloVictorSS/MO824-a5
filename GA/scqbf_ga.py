from .scqbf_instance import *
from .scqbf_solution import *
from .scqbf_evaluator import *
import time
import random
from dataclasses import dataclass
import numpy as np

from typing import List, Literal


# Type aliases:
Chromosome = List[int]  # Example chromosome representation
Population = List[Chromosome]


@dataclass
class GAStrategy:
    population_init: Literal["random", "latin_hypercube"] = "random"
    mutation_strategy: Literal["standard", "adaptive"] = "standard"
class ScQbfGA:
    
    def __init__(self, instance: ScQbfInstance, population_size: int = 100, mutation_rate_multiplier: int = 1,
                 ga_strategy: GAStrategy = GAStrategy(), termination_options: dict = {}, debug_options: dict = {}):
        # GA related properties
        self.instance = instance
        self.evaluator = ScQbfEvaluator(instance)
        self.population: Population = []
        self.best_chromosome: Chromosome = None
        self.best_solution: ScQbfSolution = None
        self.population_size = population_size
        self.mutation_rate_multiplier = mutation_rate_multiplier

        self.ga_strategy = ga_strategy
        self.debug_options = debug_options
        self.termination_options = termination_options
        self._cache = {}
        
        # Internal properties for managing execution and termination criteria
        self._start_time = None                         # Start time of the algorithm
        self.execution_time = 0.0                       # Total execution time
        self._iter = 0                                  # Current iteration
        self._no_improvement_iter = 0                   # Iterations since last improvement
        self._prev_best_solution = None                 # Previous best solution to track improvements
        self.stop_reason = None                         # Reason for stopping the algorithm (e.g., max_iter, time_limit, etc.)
        self.history: list[tuple[float, float]] = []    # History of best and current solutions' objective values
        

    def _eval_termination_condition(self) -> bool:
        """ Check if the termination condition is met, while also managing termination criteria properties."""
        self._iter += 1
        self.execution_time = time.time() - self._start_time

        max_iter = self.termination_options.get("max_iter", None)
        time_limit_secs = self.termination_options.get("time_limit_secs", None)
        patience = self.termination_options.get("patience", None)

        if max_iter is not None and self._iter > max_iter:
            self.stop_reason = "max_iter"
            return True
        if time_limit_secs is not None and self.execution_time > time_limit_secs:
            self.stop_reason = "time_limit"
            return True
        if patience is not None:
            if self._no_improvement_iter > patience:
                self.stop_reason = "patience_exceeded"
                return True
            elif self.best_solution is not None and self._prev_best_solution is not None:
                if self.evaluator.evaluate_objfun(self.best_solution) <= self.evaluator.evaluate_objfun(self._prev_best_solution):
                    self._no_improvement_iter += 1
                else:
                    self._no_improvement_iter = 0

        self._prev_best_solution = self.best_solution

        return False
    
    def _perform_debug_actions(self):
        """ Perform debug actions, such as logging or printing debug information. """
        if self.debug_options.get("verbose", False):
            print(f"Iteration {self._iter}: Best fitness = {f'{self.evaluator.evaluate_objfun(self.best_solution):.2f}' if self.best_solution else 'N/A'}")

        if self.debug_options.get("save_history", False):
            self.history.append((self.evaluator.evaluate_objfun(self.best_solution) if self.best_solution else 0))
        
        if self.debug_options.get("save_mrate_history", False):
            if not hasattr(self, 'mutation_rate_history'):
                self.mutation_rate_history = []
                self.diversity_history = []
            self.mutation_rate_history.append(self.mutation_rate_multiplier)
            self.diversity_history.append(self._get_diversity(self.population))

    def solve(self) -> ScQbfSolution:
        """ Main method to solve the problem using a genetic algorithm. """
        self._initialize_population()
        target_reached = 0
        time_to_target = -1.0
        
        self._start_time = time.time()
        self._iter = 0
        while not self._eval_termination_condition():
            self._perform_debug_actions()
            
            parents = self._select_parents()
            offspring = self._crossover(parents)
            offspring = self._mutate(offspring)
            self.population = self._select_population(offspring)

            for chromosome in self.population:
                solution, fitness = self._cached_decode_and_eval(chromosome)
                if self.best_solution is None or fitness > self.evaluator.evaluate_objfun(self.best_solution):
                    self.best_solution = solution
                    self.best_chromosome = chromosome

                    if self.best_solution.value >= self.instance.target_value:
                        time_to_target = time.time() - self._start_time
                        target_reached = 1
    
        return self.best_solution, self.execution_time, time_to_target, target_reached
        
    def decode(self, chromosome: Chromosome) -> ScQbfSolution:
        """ Decode a chromosome into a ScQbfSolution. """
        solution = ScQbfSolution([])
        for i in range(len(chromosome)):
            if chromosome[i] == 1:
                solution.elements.append(i)

        return solution

    def _cached_decode_and_eval(self, chromosome: Chromosome) -> tuple[ScQbfSolution, float]:
        key = tuple(chromosome)  # O(n) to create tuple
        if key not in self._cache:  # O(1) average lookup
            solution = self.decode(chromosome)
            fitness = self.evaluator.evaluate_objfun(solution)
            self._cache[key] = (solution, fitness)  # O(1) average insertion
        return self._cache[key]

    def _initialize_population(self):
        if self.ga_strategy.population_init == "random":
            self._initialize_population_random()
        elif self.ga_strategy.population_init == "latin_hypercube":
            self._initialize_population_latin_hypercube()
        else:
            raise ValueError(f"Unknown population initialization strategy: {self.ga_strategy.population_init}")
        
    def _initialize_population_random(self):
        self.population = []
        for _ in range(self.population_size):
            chromosome = [0] * self.instance.n
            num_ones = random.randint(1, self.instance.n)
            ones_indices = random.sample(range(self.instance.n), num_ones)
            for idx in ones_indices:
                chromosome[idx] = 1
            chromosome = self._make_feasible(chromosome)
            self.population.append(chromosome)

    def _initialize_population_latin_hypercube(self):
        if self.population_size % 2 != 0:
            raise ValueError("Population size must be a multiple of the allele count (here, 2) for Latin Hypercube Sampling.")
        
        self.population = []
        
        for i in range(self.population_size):
            chromosome = [0] * self.instance.n
            self.population.append(chromosome)
        
        # For each gene position (column), create a random permutation
        for gene_pos in range(self.instance.n):
            # Create permutation of population indices [0, 1, 2, ..., population_size-1]
            permutation = list(range(self.population_size))
            random.shuffle(permutation)
            
            # Assign alleles based on permutation index modulo 2
            for pop_idx in range(self.population_size):
                allele = permutation[pop_idx] % 2
                self.population[pop_idx][gene_pos] = allele
        
        # fix feasibility
        for i in range(len(self.population)):
            self.population[i] = self._make_feasible(self.population[i])

    def _make_feasible(self, chromosome: Chromosome) -> Chromosome:
        """
        If the chromosome is not feasible, add random elements that improve coverage until it becomes feasible.
        """
        
        decoded_solution, _ = self._cached_decode_and_eval(chromosome)
        while not self.evaluator.is_solution_feasible(decoded_solution):
            cl = [i for i in range(self.instance.n) if chromosome[i] == 0 and self.evaluator.evaluate_insertion_delta_coverage(i, decoded_solution) > 0]
            chosen = random.choice(cl)
            chromosome[chosen] = 1
            decoded_solution.elements.append(chosen)
            
        return chromosome

    def _select_parents(self) -> Population:
        """ Tournament selection implementation."""
        parents: Population = []
        
        while len(parents) < self.population_size:
            tournament = random.sample(self.population, 2)
            tournament_fitness = [self._cached_decode_and_eval(chrom)[1] for chrom in tournament]
            winner = tournament[tournament_fitness.index(max(tournament_fitness))]
            parents.append(winner)
        
        return parents
    
    def _crossover(self, parents: Population) -> Population:
        """
        Two-point crossover implementation.
        """
        offspring: Population = []
        
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            
            point1, point2 = sorted([
                random.randint(0, self.instance.n - 1),
                random.randint(0, self.instance.n - 1)
            ])
            
            offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

            offspring1 = self._make_feasible(offspring1)
            offspring2 = self._make_feasible(offspring2)

            offspring.extend([offspring1, offspring2])
        
        return offspring
    
    def _mutate(self, offspring: Population) -> Population:
        if self.ga_strategy.mutation_strategy == "standard":
            return self._mutate_standard(offspring)
        elif self.ga_strategy.mutation_strategy == "adaptive":
            return self._mutate_adaptive(offspring)
        else:
            raise ValueError(f"Unknown mutation strategy: {self.ga_strategy.mutation_strategy}")

    def _mutate_standard(self, offspring: Population) -> Population:
        """ Standard bit-flip mutation implementation using Poisson distribution."""
        for chromosome in offspring:
            lambda_param = self.mutation_rate_multiplier # expected number of mutations per chromosome
            num_mutations = np.random.poisson(lam=lambda_param)
            
            if num_mutations > 0:
                num_mutations = min(num_mutations, self.instance.n)  # Cap at chromosome length
                mutation_loci = random.sample(range(self.instance.n), num_mutations)
                
                # Flip bits at selected loci
                for locus in mutation_loci:
                    chromosome[locus] = 1 - chromosome[locus]
            
            chromosome = self._make_feasible(chromosome)
        
        return offspring
    
    def _mutate_adaptive(self, offspring: Population) -> Population:
        if not hasattr(self, 'original_mutation_rate_multiplier'):
            self.original_mutation_rate_multiplier = self.mutation_rate_multiplier
        
        if self._iter % 5 == 0:  # Adjust every 5 iterations
            if self._get_diversity(offspring) < 0.2:
                self.mutation_rate_multiplier = min(self.original_mutation_rate_multiplier + 2, self.mutation_rate_multiplier + 0.25)
            elif self._get_diversity(offspring) > 0.5:
                self.mutation_rate_multiplier = max(min(self.original_mutation_rate_multiplier - 2, 0.25), self.mutation_rate_multiplier - 0.25)

            if self.debug_options.get("verbose", False):
                print(f"Adaptive mutation rate multiplier adjusted to: {self.mutation_rate_multiplier}")
        
        return self._mutate_standard(offspring)
    
    def _get_diversity(self, population: Population) -> float:
        ''' Calculates the normalized diversity of the population.'''
        P = len(population)

        diversity_sum = 0.0
        for locus in range(self.instance.n):
            ones = sum(ch[locus] for ch in population)  # count of 1s at locus
            p = ones / P
            diversity_sum += p * (1 - p)
        
        return (diversity_sum / (0.25 * self.instance.n))

    def _select_population(self, offspring: Population) -> Population:
        """Elitist selection implementation. replace worst single offspring with best from previous generation."""
        worst_chromosome = min(offspring, 
                            key=lambda chrom: self._cached_decode_and_eval(chrom)[1])
        
        # Only replace if the worst offspring is worse than the best from previous generation
        if (self.best_chromosome is not None and 
            self._cached_decode_and_eval(worst_chromosome)[1] < 
            self._cached_decode_and_eval(self.best_chromosome)[1]):
            
            worst_index = offspring.index(worst_chromosome)
            offspring[worst_index] = self.best_chromosome
        
        return offspring

