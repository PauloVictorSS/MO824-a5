import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(csv_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(csv_path)
        print(f"Colunas encontradas: {', '.join(data.columns)}")
        return data
    except FileNotFoundError:
        print(f"ERRO: Arquivo CSV não encontrado em '{csv_path}'")
        return None
    except Exception as e:
        print(f"ERRO ao ler o arquivo CSV: {e}")
        return None

def generate_ttt_plots(df: pd.DataFrame, output_dir: str):
    if df.empty:
        return

    experiments = df[['instance', 'target_value']].drop_duplicates()
    for _, (instance_name, target_value) in experiments.iterrows():
        plt.figure(figsize=(10, 6))
        
        exp_df = df[
            (df['instance'] == instance_name) & 
            (df['target_value'] == target_value)
        ]
        
        algorithms = exp_df['algorithm'].unique()
        
        for alg in algorithms:
            alg_data = exp_df[
                (exp_df['algorithm'] == alg) & 
                (exp_df['target_reached'] == True)
            ]
            
            if alg_data.empty:
                print(f"  - Aviso: Algoritmo '{alg}' nunca atingiu o alvo para esta instância.")
                continue

            sorted_times = np.sort(alg_data['time_to_target'])
            num_runs = len(sorted_times)
            
            probabilities = (np.arange(1, num_runs + 1)) / num_runs
            
            plot_times = np.insert(sorted_times, 0, 0)
            plot_probs = np.insert(probabilities, 0, 0)
            
            plt.plot(plot_times, plot_probs, marker='.', linestyle='--', label=alg)
        
        plt.title(f"TTT-Plot para {instance_name}\nAlvo = {target_value}")
        plt.xlabel("Tempo (segundos) - Escala de Log")
        plt.ylabel("Probabilidade de Atingir o Alvo")
        plt.xscale('log') 
        plt.grid(True, which="both", linestyle=':', linewidth=0.6)
        plt.legend()
        
        filename = f"ttt_plot_{instance_name.split('.')[0]}_target_{target_value}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

def generate_performance_profile(df: pd.DataFrame, output_dir: str):
    best_per_run = df.groupby(['instance', 'algorithm'])['final_solution_value'].max().reset_index()

    best_overall = best_per_run.groupby('instance')['final_solution_value'].max().reset_index()
    best_overall = best_overall.rename(columns={'final_solution_value': 'best_value_overall'})

    merged_data = pd.merge(best_per_run, best_overall, on='instance')
    merged_data['ratio'] = merged_data['best_value_overall'] / merged_data['final_solution_value']
    
    merged_data['ratio'] = merged_data['ratio'].replace([np.inf, -np.inf], np.nan)
    
    merged_data.loc[(merged_data['best_value_overall'] == 0) & (merged_data['final_solution_value'] == 0), 'ratio'] = 1
    merged_data = merged_data.fillna(np.inf) 

    algorithms = merged_data['algorithm'].unique()
    max_ratio = merged_data[merged_data['ratio'] != np.inf]['ratio'].max()
    
    tau_range = np.linspace(1, max(2.0, max_ratio), 100)
    
    plt.figure(figsize=(10, 6))
    
    for alg in algorithms:
        alg_ratios = merged_data[merged_data['algorithm'] == alg]['ratio']
        
        probabilities = [
            (alg_ratios <= tau).mean() for tau in tau_range
        ]
        
        plt.plot(tau_range, probabilities, label=alg)

    plt.title("Perfil de Desempenho (Métrica: Melhor Solução)")
    plt.xlabel("Taxa de Desempenho (τ)")
    plt.ylabel("P(desempenho ≤ τ)")
    plt.grid(True, linestyle=':', linewidth=0.6)
    plt.legend(loc='lower right')
    plt.xlim(1, tau_range.max())
    plt.ylim(0, 1.05)
    
    filename = "performance_profile.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print("Gráfico de Perfil de Desempenho salvo.")


def main():
    CSV_FILE = 'ttt_plot_results.csv' 
    OUTPUT_DIR = 'graficos_analise'  
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Diretório '{OUTPUT_DIR}' criado.")

    data = load_data(CSV_FILE)
    
    if data is not None:
        metaheuristic_data = data[data['algorithm'] != 'PLI'].copy()
        
        if not metaheuristic_data.empty:
            generate_ttt_plots(metaheuristic_data, OUTPUT_DIR)
        else:
            print("Nenhum dado de meta-heurística encontrado para os TTT-Plots.")

        generate_performance_profile(data, OUTPUT_DIR)
        
        print("\nAnálise concluída!")
        print(f"Todos os gráficos foram salvos no diretório: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()