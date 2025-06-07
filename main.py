from tsp import *
import multiprocessing as mp
import json
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive
import matplotlib.pyplot as plt

def run_algorithm(args):
    name, func, G, coords, init_tour, params, start_node = args
    if name == "Genetic Algorithms":
        best, _, hist = func(G, **params, start=start_node)
    elif name == "AR-ACO":
        best, _, hist = func(G, coords, init_tour, **params, start=start_node)
    elif name == "Smooth ACO":
        best, _, hist = func(G, coords, init_tour, **params, start=start_node)
    elif name == "MST+AR-ACO":
        print(f"Calling {name} with args: G={G}, coords={coords is not None}, init_tour={init_tour is not None}, params keys={params.keys()}, start_node={start_node}")
        best, _, hist = func(graph=G, coords=coords, init_tour=init_tour, start=start_node, **params)
    else:
        best, _, hist = func(G, init_tour, **params, start=start_node)
    routes = split_tour_among_robots(best, num_robots, start=start_node)
    total = sum(tour_length(G, r) for r in routes)
    maximum = max(tour_length(G, r) for r in routes)
    return name, routes, total, maximum, hist

if __name__ == "__main__":
    # 参数句柄
    params = {
        "christofides_ACO": {
            "num_ants": 10,
            "num_iterations": 10,
            "alpha": 1.0,
            "beta": 3.0,
            "evaporation_rate": 0.3,
            "Q": 100
        },
        "pure_ACO": {
            "num_ants": 10,
            "num_iterations": 10,
            "alpha": 1.0,
            "beta": 3.0,
            "evaporation_rate": 0.3,
            "Q": 100
        },
        "SA": {
            "initial_temp": 1000,
            "cooling_rate": 0.995,
            "iterations": 100
        },
        "GA": {
            "population_size": 50,
            "generations": 20,
            "mutation_rate": 0.1
        }
    }
    
    num_robots = 4
    start_coords = (50, 50)
    start_node = "start"
    
    # 生成图
    G, coords = create_random_graph(200, pattern="uniform", seed=505)
    G, coords = add_start_node(G, coords, start_coords, start=start_node)
    
    # 准备算法参数
    init_tour_christofides = christofides_tsp(G, start=start_node)
    random_init = random_tour(G, start=start_node)
    
    # 定义要运行的算法
    algorithms_to_run = [
        ("MST+ACO", ant_colony_tsp, G, coords, init_tour_christofides, params["christofides_ACO"], start_node),
        ("ACO", ant_colony_tsp, G, coords, random_init, params["pure_ACO"], start_node),
        ("Simulated Annealing", simulated_annealing_tsp, G, coords, random_init, params["SA"], start_node),
        ("Genetic Algorithms", genetic_algorithm_tsp, G, coords, None, {
            "population_size": params["GA"]["population_size"],
            "generations": params["GA"]["generations"],
            "mutation_rate": params["GA"]["mutation_rate"]
        }, start_node),
        ("AR-ACO", ant_colony_AR_ACO_tsp, G, coords, random_init, 
         {"num_ants": 20, "num_iterations": 100, "alpha": 1.0, "beta": 3.0, 
          "evaporation_rate": 0.3, "Q": 100, "k": 200}, start_node),
        ("MST+AR-ACO", ant_colony_MST_AR_ACO_tsp, G, coords, random_init,
         {"num_ants": 25, "num_iterations": 100, "alpha": 1.0, "beta": 3.0,
          "evaporation_rate": 0.3, "Q": 100, "k": 200}, start_node),
        ("IEACO", ant_colony_IEACO_tsp, G, coords, random_init,
         {"num_ants": 10, "num_iterations": 100, "alpha_init": 1.0, "beta_init": 3.0,
          "evaporation_rate": 0.3, "Q": 100, "epsilon": 0.2}, start_node),
        ("DL-ACO", ant_colony_DL_ACO_tsp, G, coords, random_init,
         {"num_ants_layer1": 5, "iters_layer1": 50, "num_ants_layer2": 5, "iters_layer2": 50,
          "alpha": 1.0, "beta": 3.0, "evaporation_rate": 0.3, "Q": 100}, start_node),
        ("Smooth ACO", ant_colony_smooth_tsp, G, coords, random_init,
         {"num_ants": 10, "num_iterations": 100, "alpha": 1.0, "beta": 3.0,
          "evaporation_rate": 0.3, "Q": 100, "angle_penalty": 2.0}, start_node)
    ]
    
    # 使用多进程运行算法
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_algorithm, algorithms_to_run)
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存结果到JSON
    results_dict = {}
    for name, routes, total, maximum, hist in results:
        results_dict[name] = {
            "total_path_length": total,
            "maximum_single_robot_path": maximum,
            "history": hist.tolist() if hasattr(hist, 'tolist') else hist
        }
    
    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)
    
    # 保存路径图
    for name, routes, _, _, _ in results:
        plt.figure(figsize=(10, 10))
        plot_multi_robot_routes(coords, routes, f"multi robot - {name}")
        plt.savefig(os.path.join(results_dir, f"{name}_routes.png"))
        plt.close()
    
    # 保存综合对比图
    names, totals, maxes = zip(*[(name, total, maximum) for name, _, total, maximum, _ in results])
    x = range(len(names))
    plt.figure(figsize=(10,5))
    plt.bar(x, totals, width=0.4, label="Total path", align="center")
    plt.bar([i+0.4 for i in x], maxes, width=0.4, label="Maximum single-robot", align="center")
    plt.xticks([i+0.2 for i in x], names, rotation=45, ha="right")
    plt.ylabel("Path Length")
    plt.title("Comparison of path lengths of various algorithms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparison.png"))
    plt.close()
    
    # 打印汇总结果
    print("\n===== 各算法路径汇总 =====")
    for name, _, total, maximum, _ in results:
        print(f"{name:15} Total path length：{total:.2f}，Maximum single-machine path：{maximum:.2f}")
    
    print(f"\n结果已保存到目录: {results_dir}")