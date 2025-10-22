import random
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import sys
from dataclasses import dataclass
import os
import concurrent.futures

class SCQBFInstance:
    def __init__(self, filepath):
        self.filepath = filepath
        self.name = filepath.split('/')[-1]
        
        try:
            with open(filepath, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]

            # 1. n: número de variáveis
            self.n = int(lines[0])

            # Inicializa a matriz de coeficientes Q como um array NumPy
            self.q = {}
            
            # Inicializa a lista de subconjuntos
            self.s = []

            # 2. Subconjuntos Si (assumindo que começam da linha 3, ou índice 2)
            # Converte os elementos para 0-based index na leitura
            for i in range(self.n):
                elements = list(map(int, lines[i + 2].split()))
                self.s.append(elements)

            # 3. Matriz de coeficientes A (triangular superior)
            line_idx = self.n + 2
            for i in range(self.n):
                row_coeffs = list(map(float, lines[line_idx + i].strip().split()))
                for j_idx in range(len(row_coeffs)):
                    j = i + j_idx
                    self.q[(i, j)] = row_coeffs[j_idx]
                    self.q[(j, i)] = row_coeffs[j_idx]
                    
        except (IOError, IndexError, ValueError) as e:
            print(f"Erro ao ler ou processar o arquivo de instância: {filepath}")
            print(f"Detalhe do erro: {e}")
            # Lança a exceção para interromper a execução se a leitura falhar
            raise

@dataclass
class SCQBFSolution:
    x: list[int]
    value: float

def scqbf_evaluate(instance: SCQBFInstance, solution) -> float:
    if hasattr(solution, 'value') and solution.value != 0:
        return solution.value
    
    objective_value = 0.0
    for (i, j), q_ij in instance.q.items():
        if solution[i] == 1 and solution[j] == 1:
            objective_value += q_ij
    
    return objective_value

def solve_pli(instance, time_limit=600):
    model = gp.Model("MAX-SC-QBF-Linearized")
    n = instance.n
    a = instance.q
    
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, n, vtype=GRB.BINARY, name="y")

    obj = gp.LinExpr()
    for i in range(n):
      for j in range(i, n):
        if (i, j) in a:
          obj += a[(i, j)] * y[i, j]
    model.setObjective(obj, GRB.MAXIMIZE)

    for i in range(n):
      for j in range(i, n):
        model.addConstr(y[i, j] <= x[i], name=f"lin_yij_le_xi_{i}_{j}")
        model.addConstr(y[i, j] <= x[j], name=f"lin_yij_le_xj_{i}_{j}")
        model.addConstr(y[i, j] >= x[i] + x[j] - 1, name=f"lin_yij_ge_sum_{i}_{j}")

    for k in range(1, n + 1):
      covering_subsets = []
      for i in range(n):
        if k in instance.s[i]:
          covering_subsets.append(i)
    
      if covering_subsets:
        model.addConstr(sum(x[i] for i in covering_subsets) >= 1, name=f"cover_{k}")

    model.setParam('TimeLimit', time_limit)
    model.setParam('OutputFlag', 0)
    start_time = time.time()
    model.optimize()
    end_time = time.time()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        solution_x = [x[i].x for i in range(n)]
        solution = SCQBFSolution(solution_x, model.objVal)
        return solution, end_time - start_time
    else:
        print("PLI: Nenhuma solução viável encontrada.")
        return SCQBFSolution(np.zeros(instance.n), -np.inf), end_time - start_time
    
def construct_greedy_randomized(instance: SCQBFInstance, alpha: float) -> list[int]:
    solution = [0] * instance.n
    
    subset_indices = list(range(instance.n))
    random.shuffle(subset_indices)

    for i in subset_indices:
        candidates = []
        for j in instance.s[i]:
            gain = instance.q.get((j, j), 0.0)
            for k in range(instance.n):
                if solution[k] == 1:
                    gain += instance.q.get(tuple(sorted((k, j))), 0.0)
            candidates.append((gain, j))

        if not candidates:
            continue

        candidates.sort(key=lambda item: item[0], reverse=True)

        best_gain = candidates[0][0]
        worst_gain = candidates[-1][0]
        
        threshold = best_gain - alpha * (best_gain - worst_gain)
        
        rcl = [cand for cand in candidates if cand[0] >= threshold]
        _, chosen_variable_idx = random.choice(rcl)
        solution[chosen_variable_idx - 1] = 1
        
    return solution

def local_search(instance: SCQBFInstance, initial_solution: SCQBFSolution) -> SCQBFSolution:
    best_solution = SCQBFSolution(initial_solution.x, initial_solution.value)
    
    improved = True
    while improved:
        improved = False
        for i in range(instance.n): 
            current_j = -1
            for j in instance.s[i]:
                if best_solution.x[j - 1] == 1:
                    current_j = j
                    break
            
            for k in instance.s[i]:
                if k == current_j:
                    continue
                
                neighbor_x = list(best_solution.x)
                neighbor_x[current_j - 1] = 0
                neighbor_x[k - 1] = 1
                
                neighbor_value = scqbf_evaluate(instance, neighbor_x)
                
                if neighbor_value > best_solution.value:
                    best_solution.x = neighbor_x
                    best_solution.value = neighbor_value
                    improved = True
                    break 
            if improved:
                break
    
    return best_solution


def solve_grasp(instance: SCQBFInstance, time_limit: int, alpha: float = 0.3, target_value=None) -> tuple:
    start_time = time.time()
    best_solution_so_far = None
    iterations = 0
    time_to_target = -1.0
    target_reached = False

    while time.time() - start_time < time_limit:
        constructed_x = construct_greedy_randomized(instance, alpha)
        constructed_value = scqbf_evaluate(instance, constructed_x)
        initial_solution = SCQBFSolution(constructed_x, constructed_value)

        if target_value is not None and initial_solution.value >= target_value:
            time_to_target = time.time() - start_time
            target_reached = True
        
        local_search_solution = local_search(instance, initial_solution)
        
        if best_solution_so_far is None or local_search_solution.value > best_solution_so_far.value:
            best_solution_so_far = local_search_solution
            if target_value is not None and best_solution_so_far.value >= target_value:
                time_to_target = time.time() - start_time
                target_reached = True

        iterations += 1

    end_time = time.time()
    if target_reached:
        return best_solution_so_far, end_time - start_time, time_to_target, True
    else:
        return best_solution_so_far, end_time - start_time, end_time - start_time, False


def get_initial_solution(instance: SCQBFInstance) -> SCQBFSolution:
    x = [0] * instance.n
    for subset in instance.s:
        if subset:
            chosen_variable = random.choice(subset)
            x[chosen_variable - 1] = 1
    
    value = scqbf_evaluate(instance, x)
    return SCQBFSolution(x, value)


def solve_tabu_search(instance: SCQBFInstance, time_limit: int, tabu_tenure: int = 10, target_value=None) -> tuple:
    start_time = time.time()
    time_to_target = -1.0
    target_reached = False
    tabu_list = {}
    
    current_solution = get_initial_solution(instance)
    best_solution_so_far = SCQBFSolution(list(current_solution.x), current_solution.value)

    if target_value is not None and best_solution_so_far.value >= target_value:
        time_to_target = time.time() - start_time
        target_reached = True

    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        best_neighbor_val = -np.inf  
        best_neighbor_x = None       
        best_neighbor_move = None   

        for i in range(instance.n): 
            current_j = -1
            for j_1_based in instance.s[i]:
                if current_solution.x[j_1_based - 1] == 1:
                    current_j = j_1_based
                    break
            
            if current_j == -1: continue 

            for k_1_based in instance.s[i]:
                if k_1_based == current_j:
                    continue
                
                neighbor_x = np.copy(current_solution.x)
                neighbor_x[current_j - 1] = 0
                neighbor_x[k_1_based - 1] = 1
                
                neighbor_value = scqbf_evaluate(instance, neighbor_x)
                
                is_tabu = tabu_list.get(current_j, 0) > iteration
                is_aspirated = neighbor_value > best_solution_so_far.value
                
                if not is_tabu or is_aspirated:
                    if neighbor_value > best_neighbor_val:
                        best_neighbor_val = neighbor_value
                        best_neighbor_x = neighbor_x
                        best_neighbor_move = (current_j, k_1_based)
        
        if best_neighbor_x is None:
            continue 
            
        current_solution = SCQBFSolution(best_neighbor_x, best_neighbor_val)
        
        removed_var, added_var = best_neighbor_move
        tabu_list[removed_var - 1] = iteration + tabu_tenure
        
        if current_solution.value > best_solution_so_far.value:
            best_solution_so_far = SCQBFSolution(current_solution.x, current_solution.value)
            if not target_reached and target_value is not None and best_solution_so_far.value >= target_value:
                time_to_target = time.time() - start_time
                target_reached = True
            
        iteration += 1

    end_time = time.time()
    if target_reached:
        return best_solution_so_far, end_time - start_time, time_to_target, True
    else:
        return best_solution_so_far, end_time - start_time, end_time - start_time, False


def create_initial_population(instance: SCQBFInstance, pop_size: int) -> list[SCQBFSolution]:
    population = []
    for _ in range(pop_size):
        x = [0] * instance.n
        for subset in instance.s:
            if subset:
                chosen_variable = random.choice(subset)
                x[chosen_variable - 1] = 1
        
        fitness = scqbf_evaluate(instance, x)
        population.append(SCQBFSolution(x, fitness))
    return population

def tournament_selection(population: list[SCQBFSolution], k: int) -> SCQBFSolution:
    tournament_contenders = random.sample(population, k)
    winner = max(tournament_contenders, key=lambda individual: individual.value)
    return winner

def crossover(instance: SCQBFInstance, parent1_x: list[int], parent2_x: list[int]) -> tuple[list[int], list[int]]:
    offspring1_x = [0] * instance.n
    offspring2_x = [0] * instance.n
    
    for subset in instance.s:
        p1_choice, p2_choice = -1, -1
        for var_idx in subset:
            if parent1_x[var_idx - 1] == 1: p1_choice = var_idx
            if parent2_x[var_idx - 1] == 1: p2_choice = var_idx
        
        if random.random() < 0.5:
            offspring1_x[p1_choice - 1] = 1
            offspring2_x[p2_choice - 1] = 1
        else:
            offspring1_x[p2_choice - 1] = 1
            offspring2_x[p1_choice - 1] = 1
            
    return offspring1_x, offspring2_x

def mutate(instance: SCQBFInstance, solution_x: list[int], mutation_rate: float) -> list[int]:
    mutated_x = list(solution_x)
    for subset in instance.s:
        if random.random() < mutation_rate and len(subset) > 1:
            current_choice = -1
            for var_idx in subset:
                if mutated_x[var_idx - 1] == 1:
                    current_choice = var_idx
                    break
            
            new_choice = random.choice([v for v in subset if v != current_choice])
            
            mutated_x[current_choice - 1] = 0
            mutated_x[new_choice - 1] = 1

    return mutated_x

def solve_genetic_algorithm(instance, time_limit=600, pop_size=40, elite_size=5, mutation_rate=0.1, tournament_size=3, crossover_rate=0.8, target_value=None):
    start_time = time.time()
    time_to_target = -1.0
    target_reached = False
    
    POP_SIZE = pop_size
    CROSSOVER_RATE = crossover_rate
    MUTATION_RATE = mutation_rate
    TOURNAMENT_SIZE = tournament_size
    ELITISM_COUNT = elite_size

    population = create_initial_population(instance, POP_SIZE)
    
    # Cria uma cópia do melhor atual
    initial_best = max(population, key=lambda ind: ind.value)
    best_solution_so_far = SCQBFSolution(initial_best.x, initial_best.value)

    generation = 0
    while time.time() - start_time < time_limit:
        population.sort(key=lambda ind: ind.value, reverse=True)
        
        new_population = []
        
        if ELITISM_COUNT > 0:
            new_population.extend(population[:ELITISM_COUNT])
        
        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population, TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, TOURNAMENT_SIZE)
            
            if random.random() < CROSSOVER_RATE:
                offspring1_x, offspring2_x = crossover(instance, parent1.x, parent2.x)
            else:
                offspring1_x, offspring2_x = list(parent1.x), list(parent2.x)
            
            offspring1_x = mutate(instance, offspring1_x, MUTATION_RATE)
            offspring2_x = mutate(instance, offspring2_x, MUTATION_RATE)
            
            new_population.append(SCQBFSolution(offspring1_x, scqbf_evaluate(instance, offspring1_x)))
            if len(new_population) < POP_SIZE:
                new_population.append(SCQBFSolution(offspring2_x, scqbf_evaluate(instance, offspring2_x)))

        population = new_population
        
        current_best = max(population, key=lambda ind: ind.value)
        if current_best.value > best_solution_so_far.value:
            best_solution_so_far.x = current_best.x
            best_solution_so_far.value = current_best.value
            if not target_reached and target_value is not None and best_solution_so_far.value >= target_value:
                time_to_target = time.time() - start_time
                target_reached = True

        generation += 1

    end_time = time.time()
    if target_reached:
        return best_solution_so_far, end_time - start_time, time_to_target, True
    else:
        return best_solution_so_far, end_time - start_time, end_time - start_time, False


def _run_single_execution(instance_file: str, alg_name: str, seed: int, time_limit: int, target_value):
    """Worker executed in a separate process: carrega a instância, seta as sementes e executa o algoritmo."""
    # Reimportante: função definida no módulo (nível superior) para ser picklable no Windows
    random.seed(seed)
    np.random.seed(seed)

    # carrega a instância dentro do worker para evitar problemas de pickle
    instance = SCQBFInstance(instance_file)

    # mapear nome do algoritmo para a função
    if alg_name == 'GRASP':
        func = solve_grasp
    elif alg_name == 'TabuSearch':
        func = solve_tabu_search
    elif alg_name == 'GeneticAlgorithm':
        func = solve_genetic_algorithm
    elif alg_name == 'PLI':
        func = solve_pli
    else:
        raise ValueError(f"Algoritmo desconhecido: {alg_name}")
    print(f"\n     [{time.strftime('%H:%M')}] Executando seed {seed} com {alg_name}...", flush=True)
    solution, total_time, time_to_target, target_reached = func(
        instance,
        time_limit=time_limit,
        target_value=target_value
    )

    return (instance.name, alg_name, seed, target_value, bool(target_reached), float(time_to_target), float(solution.value), float(total_time))


if __name__ == '__main__':
    TTT_PLOT_CONFIG = {
        'in/instance-05.txt': 1100,
        'in/instance-10.txt': 4000,
        'in/instance-15.txt': 8000,
    }

    # 2. Parâmetros do experimento
    NUM_EXECUTIONS = 20
    TIME_LIMIT_SECONDS = 600 
    OUTPUT_FILE = 'ttt_plot_results.csv'
    
    # 3. Algoritmos a serem testados
    algorithms = {
        #'GRASP': solve_grasp,
        'TabuSearch': solve_tabu_search,
        'GeneticAlgorithm': solve_genetic_algorithm
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        # Escreve o cabeçalho do arquivo CSV
        f.write("instance,algorithm,execution_seed,target_value,target_reached,time_to_target,final_solution_value,total_time\n")

        for instance_file, target_value in TTT_PLOT_CONFIG.items():
            print(f"\n--- Processando Instância: {instance_file} (Alvo: {target_value}) ---")

            for alg_name, solve_function in algorithms.items():
                print(f"  -> Executando Algoritmo: {alg_name}")

                # cria tarefas para todos os seeds
                tasks = [
                    (instance_file, alg_name, seed, TIME_LIMIT_SECONDS, target_value)
                    for seed in range(NUM_EXECUTIONS)
                ]

                max_workers = max(min(NUM_EXECUTIONS, os.cpu_count()) - 2, 1)

                # executa em paralelo e coleta resultados; grava no CSV no processo principal
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_run_single_execution, *t): t[2] for t in tasks}
                    completed = 0
                    for fut in concurrent.futures.as_completed(futures):
                        completed += 1
                        try:
                            (inst_name, alg, seed, tgt, reached, t_to_tgt, sol_val, tot_time) = fut.result()
                        except Exception as e:
                            print(f"\n    [{time.strftime('%H:%M')}] Erro na execução do seed {futures[fut]}: {e}")
                            continue

                        # Salva os resultados (serializado no processo principal)
                        f.write(
                            f"{inst_name},"
                            f"{alg},"
                            f"{seed},"
                            f"{tgt},"
                            f"{int(reached)},"
                            f"{t_to_tgt:.4f},"
                            f"{sol_val:.4f},"
                            f"{tot_time:.4f}\n"
                        )

                        print(f"\r     [{time.strftime('%H:%M')}] Execuções concluídas: {completed}/{NUM_EXECUTIONS}...", end="")

                print(f"\n[{time.strftime('%H:%M')}]...Concluído.")

    print(f"\n     [{time.strftime('%H:%M')}] Experimento finalizado! Resultados salvos em '{OUTPUT_FILE}'.")