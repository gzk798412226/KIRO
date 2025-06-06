from tsp import *

if __name__ == "__main__":
    # 参数句柄
    params = {
        "christofides_ACO": {
            "num_ants": 10,
            "num_iterations": 50,
            "alpha": 1.0,
            "beta": 3.0,
            "evaporation_rate": 0.3,
            "Q": 100
        },
        "pure_ACO": {
            "num_ants": 10,
            "num_iterations": 50,
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
    
    # 1. Christofides + ACO
    init_tour_christofides = christofides_tsp(G, start=start_node)
    best_ca, _, hist_ca = ant_colony_tsp(
        G, init_tour_christofides,
        **params["christofides_ACO"], start=start_node
    )
    routes_ca = split_tour_among_robots(best_ca, num_robots, start=start_node)
    
    # 2. 纯 ACO
    random_init = random_tour(G, start=start_node)
    best_aco, _, hist_aco = ant_colony_tsp(
        G, random_init,
        **params["pure_ACO"], start=start_node
    )
    routes_aco = split_tour_among_robots(best_aco, num_robots, start=start_node)
    
    # 3. 模拟退火
    sa_init = random_tour(G, start=start_node)
    best_sa, _, hist_sa = simulated_annealing_tsp(
        G, sa_init,
        **params["SA"], start=start_node
    )
    routes_sa = split_tour_among_robots(best_sa, num_robots, start=start_node)
    
    # 4. 遗传算法
    best_ga, _, hist_ga = genetic_algorithm_tsp(
        G,
        population_size=params["GA"]["population_size"],
        generations=params["GA"]["generations"],
        mutation_rate=params["GA"]["mutation_rate"],
        start=start_node
    )
    routes_ga = split_tour_among_robots(best_ga, num_robots, start=start_node)
    
    # 5. AR-ACO
    best_ar, _, hist_ar = ant_colony_AR_ACO_tsp(
        G, coords, random_init,
        num_ants=20, num_iterations=100,
        alpha=1.0, beta=3.0,
        evaporation_rate=0.3, Q=100,
        start=start_node, k=200
    )
    routes_ar = split_tour_among_robots(best_ar, num_robots, start=start_node)
    
    # 6. IEACO
    best_ie, _, hist_ie = ant_colony_IEACO_tsp(
        G, random_init,
        num_ants=10, num_iterations=100,
        alpha_init=1.0, beta_init=3.0,
        evaporation_rate=0.3, Q=100,
        epsilon=0.2, start=start_node
    )
    routes_ie = split_tour_among_robots(best_ie, num_robots, start=start_node)
    
    # 7. DL-ACO
    best_dl, _, hist_dl = ant_colony_DL_ACO_tsp(
        G, random_init,
        num_ants_layer1=5, iters_layer1=50,
        num_ants_layer2=5, iters_layer2=50,
        alpha=1.0, beta=3.0,
        evaporation_rate=0.3, Q=100,
        start=start_node
    )
    routes_dl = split_tour_among_robots(best_dl, num_robots, start=start_node)
    
    # 8. 平滑ACO
    best_sm, _, hist_sm = ant_colony_smooth_tsp(
        G, coords, random_init,
        num_ants=10, num_iterations=100,
        alpha=1.0, beta=3.0,
        evaporation_rate=0.3, Q=100,
        start=start_node, angle_penalty=2.0
    )
    routes_sm = split_tour_among_robots(best_sm, num_robots, start=start_node)
    

# 汇总打印
algorithms = [
    ("MST+ACO", routes_ca),
    ("ACO", routes_aco),
    ("Simulated Annealing", routes_sa),
    ("Genetic Algorithms", routes_ga),
    ("AR-ACO", routes_ar),
    ("IEACO", routes_ie),
    ("DL-ACO", routes_dl),
    ("Smooth ACO", routes_sm),
]
print("\n===== 各算法路径汇总 =====")
summary = []
for name, routes in algorithms:
    total = sum(tour_length(G, r) for r in routes)
    maximum = max(tour_length(G, r) for r in routes)
    print(f"{name:15} Total path length：{total:.2f}，Maximum single-machine path：{maximum:.2f}")
    summary.append((name, total, maximum))

# 静态路径图
for name, routes in algorithms:
    plot_multi_robot_routes(coords, routes, f"multi robot - {name}")

# 综合条形图对比
import matplotlib.pyplot as plt
names, totals, maxes = zip(*summary)
x = range(len(names))
plt.figure(figsize=(10,5))
plt.bar(x, totals, width=0.4, label="Total path", align="center")
plt.bar([i+0.4 for i in x], maxes, width=0.4, label="Maximum single-robot", align="center")
plt.xticks([i+0.2 for i in x], names, rotation=45, ha="right")
plt.ylabel("Path Length")
plt.title("Comparison of path lengths of various algorithms")
plt.legend()
plt.tight_layout()
plt.show()