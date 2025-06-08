import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_experiment_results(results_dir):
    """Load data from a single experiment result directory"""
    with open(os.path.join(results_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    with open(os.path.join(results_dir, "results.json"), "r") as f:
        results = json.load(f)
    
    return config, results

def analyze_results():
    # Get all result directories
    results_dirs = sorted(glob.glob("results_*"))
    
    # Prepare data frame
    data = []
    
    for dir_path in results_dirs:
        try:
            config, results = load_experiment_results(dir_path)
            
            for algo_name, algo_results in results.items():
                data.append({
                    "timestamp": config["timestamp"],
                    "seed": config["seed"],
                    "num_cities": config["num_cities"],
                    "num_robots": config["num_robots"],
                    "start_x": config["start_coords"][0],
                    "start_y": config["start_coords"][1],
                    "algorithm": algo_name,
                    "total_path_length": algo_results["total_path_length"],
                    "maximum_single_robot_path": algo_results["maximum_single_robot_path"]
                })
        except Exception as e:
            print(f"Error processing {dir_path}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Create analysis directory
    analysis_dir = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 1. Scale Analysis
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="num_cities", y="total_path_length", hue="algorithm")
    plt.title("Total Path Length vs Problem Scale")
    plt.xlabel("Number of Cities")
    plt.ylabel("Total Path Length")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "scale_analysis.png"))
    plt.close()
    
    # 2. Robot Number Analysis
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="num_robots", y="maximum_single_robot_path", hue="algorithm")
    plt.title("Maximum Single Robot Path Length vs Number of Robots")
    plt.xlabel("Number of Robots")
    plt.ylabel("Maximum Single Robot Path Length")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "robot_analysis.png"))
    plt.close()
    
    # 3. Start Position Analysis
    plt.figure(figsize=(12, 6))
    df['start_position'] = df['start_x'].astype(str) + ',' + df['start_y'].astype(str)
    sns.boxplot(data=df, x="start_position", y="total_path_length", hue="algorithm")
    plt.title("Total Path Length vs Start Position")
    plt.xlabel("Start Position (x,y)")
    plt.ylabel("Total Path Length")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, "start_position_analysis.png"))
    plt.close()
    
    # 4. Algorithm Performance Statistics
    stats = df.groupby('algorithm').agg({
        'total_path_length': ['mean', 'std', 'min', 'max'],
        'maximum_single_robot_path': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    # Save statistics
    stats.to_csv(os.path.join(analysis_dir, "algorithm_stats.csv"))
    
    # 5. Generate Detailed Report
    with open(os.path.join(analysis_dir, "analysis_report.txt"), "w") as f:
        f.write("=== Experiment Analysis Report ===\n\n")
        
        f.write("1. Experiment Configuration Statistics\n")
        f.write(f"Total number of experiments: {len(results_dirs)}\n")
        f.write(f"City count range: {df['num_cities'].min()} - {df['num_cities'].max()}\n")
        f.write(f"Robot count range: {df['num_robots'].min()} - {df['num_robots'].max()}\n")
        f.write(f"Start position range: X({df['start_x'].min()}-{df['start_x'].max()}), Y({df['start_y'].min()}-{df['start_y'].max()})\n\n")
        
        f.write("2. Algorithm Performance Statistics\n")
        f.write(stats.to_string())
        f.write("\n\n")
        
        f.write("3. Key Findings\n")
        # Find algorithm with shortest total path length
        best_total = df.loc[df.groupby('num_cities')['total_path_length'].idxmin()]
        f.write("Algorithms with shortest total path length:\n")
        f.write(best_total[['num_cities', 'algorithm', 'total_path_length']].to_string())
        f.write("\n\n")
        
        # Find algorithm with shortest maximum single robot path
        best_max = df.loc[df.groupby('num_robots')['maximum_single_robot_path'].idxmin()]
        f.write("Algorithms with shortest maximum single robot path:\n")
        f.write(best_max[['num_robots', 'algorithm', 'maximum_single_robot_path']].to_string())
    
    print(f"Analysis complete. Results saved in directory: {analysis_dir}")

if __name__ == "__main__":
    analyze_results() 