import concurrent.futures
import os
import random
import time
import numpy as np

## import solvers ##
from GA.scqbf_instance import *
from GA.scqbf_solution import *
from GA.scqbf_evaluator import *
from GA.scqbf_ga import *

from GRASP.src.grasp_maxsc_qbf.algorithms.grasp_qbf_sc import *

## Arguments ##
# GRASP parameters
alpha = 0.1
iterations = 1000
# Genetic Algorithm parameters
STRAT = GAStrategy(population_init='latin_hypercube', mutation_strategy='standard')
MUT_RATE = 1
POP_SIZE = 50
## Solvers ##

def solve_grasp(instance_file, time_limit, target_value):
    grasp = GRASP_QBF_SC(alpha=alpha, iterations=iterations, filename=instance_file)
    best_solution, elapsed_time, time_to_target, target_reached = grasp.solve(time_limit, target_value)

    return best_solution, elapsed_time, time_to_target, target_reached

def solve_tabu_search(instance_file, time_limit, target_value):
    pass

def solve_ga(instance_file, time_limit, target_value):
    instance = read_max_sc_qbf_instance(instance_file, target_value)
    ga = ScQbfGA(instance, POP_SIZE, MUT_RATE, termination_options={'time_limit_secs': time_limit}, ga_strategy=STRAT)
    best_solution, elapsed_time, time_to_target, target_reached = ga.solve()

    evaluator = ScQbfEvaluator(instance)
    evaluator.evaluate_objfun(best_solution)

    return best_solution, elapsed_time, time_to_target, target_reached



def solve_pli(instance_file, time_limit, target_value):
    pass

# Utility function

def _run_single_execution(instance_file: str, alg_name: str, seed: int, time_limit: int, target_value):
    """Worker executed in a separate process: carrega a instância, seta as sementes e executa o algoritmo."""
    # Reimportante: função definida no módulo (nível superior) para ser picklable no Windows
    random.seed(seed)
    np.random.seed(seed)

    # mapear nome do algoritmo para a função
    if alg_name == 'GRASP':
        func = solve_grasp
    elif alg_name == 'TabuSearch':
        func = solve_tabu_search
    elif alg_name == 'GeneticAlgorithm':
        func = solve_ga
    elif alg_name == 'PLI':
        func = solve_pli
    else:
        raise ValueError(f"Algoritmo desconhecido: {alg_name}")
    print(f"\n     [{time.strftime('%H:%M')}] Executando seed {seed} com {alg_name}...", flush=True)
    solution, total_time, time_to_target, target_reached = func(
        instance_file,
        time_limit=time_limit,
        target_value=target_value
    )

    return (instance_file, alg_name, seed, target_value, bool(target_reached), float(time_to_target), float(solution.value), float(total_time))


if __name__ == '__main__':
    TTT_PLOT_CONFIG = {
        'in/instance-05.txt': 1100,
        'in/instance-10.txt': 4000,
        'in/instance-15.txt': 8000,
    }

    # 2. Parâmetros do experimento
    NUM_EXECUTIONS = 20
    TIME_LIMIT_SECONDS = 10 * 60  # 10 minutos por execução 
    OUTPUT_FILE = 'ttt_plot_results.csv'
    
    # 3. Algoritmos a serem testados
    algorithms = {
        'GRASP': solve_grasp,
        #'TabuSearch': solve_tabu_search,
        #'GeneticAlgorithm': solve_ga
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
