import networkx as nx
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# -------------------------------
# 1. 斥力场 ACO（AR-ACO）
# -------------------------------
def compute_repulsive_field(coords, start="start", k=100):
    """针对每个节点计算到 start 的斥力（距离越近斥力越大）"""
    repulsive = {}
    sx, sy = coords[start]
    for node, (x, y) in coords.items():
        if node == start:
            repulsive[node] = 0.0
        else:
            d = math.hypot(x-sx, y-sy)
            repulsive[node] = k / (d**2 + 1e-6)
    return repulsive

def construct_tour_AR(graph, nodes, pheromone, alpha, beta, repulsive, start="start"):
    current = start
    tour = [current]
    unvisited = set(nodes) - {start}
    while unvisited:
        probs = []
        total = 0.0
        for nxt in unvisited:
            tau = pheromone[(current, nxt)]
            base_eta = 1.0 / graph[current][nxt]['weight']
            # 将斥力场加入启发式
            eta = base_eta / (1 + repulsive[nxt])
            p = (tau**alpha) * (eta**beta)
            probs.append((nxt, p))
            total += p
        probs = [(n, p/total) for n, p in probs]
        r = random.random()
        cum = 0.0
        for n, p in probs:
            cum += p
            if r <= cum:
                choice = n
                break
        tour.append(choice)
        unvisited.remove(choice)
        current = choice
    return tour

def ant_colony_AR_ACO_tsp(graph, coords, init_tour,
                          num_ants, num_iterations,
                          alpha, beta, evaporation_rate, Q,
                          start="start", k=100):
    nodes = list(graph.nodes())
    # 初始化信息素
    pheromone = {(u,v):1.0 for u,v,_ in graph.edges(data=True)}
    pheromone.update({(v,u):1.0 for u,v,_ in graph.edges(data=True)})
    repulsive = compute_repulsive_field(coords, start, k)

    best_tour = init_tour[:]
    best_len = tour_length(graph, best_tour)
    history = [best_tour.copy()]

    for it in range(num_iterations):
        all_tours, all_lens = [], []
        for _ in range(num_ants):
            tour = construct_tour_AR(graph, nodes, pheromone, alpha, beta, repulsive, start)
            L = tour_length(graph, tour)
            all_tours.append(tour); all_lens.append(L)
            if L < best_len:
                best_len = L; best_tour = tour.copy()
        # 挥发
        for key in pheromone:
            pheromone[key] *= (1 - evaporation_rate)
        # 更新
        for tour, L in zip(all_tours, all_lens):
            d = Q / L
            for i in range(len(tour)-1):
                u, v = tour[i], tour[i+1]
                pheromone[(u,v)] += d
                pheromone[(v,u)] += d
        history.append(best_tour.copy())

    return best_tour, best_len, history


# -------------------------------
# 2. 智能增强蚁群（IEACO）
# -------------------------------
def ant_colony_IEACO_tsp(graph, init_tour,
                         num_ants, num_iterations,
                         alpha_init, beta_init,
                         evaporation_rate, Q,
                         epsilon=0.1, start="start"):
    nodes = list(graph.nodes())
    pheromone = {(u,v):1.0 for u,v,_ in graph.edges(data=True)}
    pheromone.update({(v,u):1.0 for u,v,_ in graph.edges(data=True)})

    best_tour = init_tour[:]
    best_len = tour_length(graph, best_tour)
    history = [best_tour.copy()]

    for it in range(num_iterations):
        # 动态调整 alpha/beta
        alpha = alpha_init * (1 + it/num_iterations)
        beta  = beta_init  * (1 - it/num_iterations)

        all_tours, all_lens = [], []
        for _ in range(num_ants):
            tour = [start]
            current = start
            unvisited = set(nodes) - {start}
            while unvisited:
                if random.random() < epsilon:
                    # 探索：随机选
                    nxt = random.choice(list(unvisited))
                else:
                    # 利用：按概率选
                    probs = []
                    total = 0.0
                    for j in unvisited:
                        tau = pheromone[(current,j)]
                        eta = 1.0/graph[current][j]['weight']
                        p = (tau**alpha)*(eta**beta)
                        probs.append((j,p)); total += p
                    probs = [(j,p/total) for j,p in probs]
                    r = random.random(); cum=0.0
                    for j,p in probs:
                        cum += p
                        if r <= cum:
                            nxt = j
                            break
                tour.append(nxt)
                unvisited.remove(nxt)
                current = nxt

            L = tour_length(graph, tour)
            all_tours.append(tour); all_lens.append(L)
            if L < best_len:
                best_len = L; best_tour = tour.copy()

        # 信息素挥发
        for key in pheromone:
            pheromone[key] *= (1 - evaporation_rate)
        # 信息素更新
        for tour, L in zip(all_tours, all_lens):
            d = Q / L
            for i in range(len(tour)-1):
                u, v = tour[i], tour[i+1]
                pheromone[(u,v)] += d
                pheromone[(v,u)] += d

        history.append(best_tour.copy())

    return best_tour, best_len, history


# -------------------------------
# 3. 双层蚁群（DL-ACO）
# -------------------------------
def ant_colony_DL_ACO_tsp(graph, init_tour,
                          num_ants_layer1, iters_layer1,
                          num_ants_layer2, iters_layer2,
                          alpha, beta, evaporation_rate, Q,
                          start="start"):
    # 第一层：PEACO —— 快速全局收敛
    best1, len1, hist1 = ant_colony_tsp(
        graph, init_tour,
        num_ants_layer1, iters_layer1,
        alpha, beta, evaporation_rate, Q, start
    )
    # 第二层：从 best1 进一步细化
    best2, len2, hist2 = ant_colony_tsp(
        graph, best1,
        num_ants_layer2, iters_layer2,
        alpha, beta, evaporation_rate, Q, start
    )
    # 合并历史：去掉第二层的第一个（与 hist1 重复）
    history = hist1 + hist2[1:]
    return best2, len2, history


# -------------------------------
# 4. 平滑路径 ACO 变种
# -------------------------------
def construct_tour_smooth(graph, coords, nodes, pheromone, alpha, beta,
                          start="start", angle_penalty=1.0):
    tour = [start]
    current = start
    prev = None
    unvisited = set(nodes) - {start}
    while unvisited:
        probs = []
        total = 0.0
        for nxt in unvisited:
            tau = pheromone[(current,nxt)]
            base_eta = 1.0/graph[current][nxt]['weight']
            if prev is None:
                penalty = 1.0
            else:
                # 计算转向角度
                x0,y0 = coords[prev]
                x1,y1 = coords[current]
                x2,y2 = coords[nxt]
                v1 = (x1-x0, y1-y0)
                v2 = (x2-x1, y2-y1)
                cosang = (v1[0]*v2[0] + v1[1]*v2[1]) / (math.hypot(*v1)*math.hypot(*v2)+1e-6)
                angle = math.acos(max(-1,min(1,cosang)))
                penalty = 1.0/(1 + angle*angle_penalty)
            eta = base_eta * penalty
            p = (tau**alpha)*(eta**beta)
            probs.append((nxt,p)); total += p
        probs = [(n,p/total) for n,p in probs]
        r = random.random(); cum = 0.0
        for n,p in probs:
            cum += p
            if r <= cum:
                choice = n
                break
        tour.append(choice)
        unvisited.remove(choice)
        prev = current
        current = choice
    return tour

def ant_colony_smooth_tsp(graph, coords, init_tour,
                          num_ants, num_iterations,
                          alpha, beta, evaporation_rate, Q,
                          start="start", angle_penalty=1.0):
    nodes = list(graph.nodes())
    pheromone = {(u,v):1.0 for u,v,_ in graph.edges(data=True)}
    pheromone.update({(v,u):1.0 for u,v,_ in graph.edges(data=True)})

    best = init_tour[:]
    best_len = tour_length(graph, best)
    history = [best.copy()]

    for it in range(num_iterations):
        all_tours, all_lens = [], []
        for _ in range(num_ants):
            tour = construct_tour_smooth(
                graph, coords, nodes, pheromone,
                alpha, beta, start, angle_penalty
            )
            L = tour_length(graph, tour)
            all_tours.append(tour); all_lens.append(L)
            if L < best_len:
                best_len = L; best = tour.copy()
        # 挥发 & 更新
        for key in pheromone:
            pheromone[key] *= (1 - evaporation_rate)
        for tour, L in zip(all_tours, all_lens):
            d = Q / L
            for i in range(len(tour)-1):
                u, v = tour[i], tour[i+1]
                pheromone[(u,v)] += d
                pheromone[(v,u)] += d
        history.append(best.copy())

    return best, best_len, history
# -------------------------------
# 辅助函数：计算开放路径总长度（只计算相邻节点距离）
# -------------------------------
def tour_length(graph, tour):
    length = 0.0
    for i in range(len(tour) - 1):
        length += graph[tour[i]][tour[i+1]]['weight']
    return length

# -------------------------------
# 辅助函数：旋转路径，使得起点为指定 start
# -------------------------------
def ensure_start(tour, start="start"):
    if tour[0] == start:
        return tour
    if start in tour:
        idx = tour.index(start)
        new_tour = tour[idx:] + tour[:idx]
        return new_tour
    return tour

# -------------------------------
# 辅助函数：将单机器人开放解均匀拆分给多个机器人
# -------------------------------
def split_tour_among_robots(tour, num_robots, start="start"):
    tour = ensure_start(tour, start)
    nodes = tour[1:]  # 去掉起点
    n = len(nodes)
    routes = []
    base = n // num_robots
    extra = n % num_robots
    index = 0
    for i in range(num_robots):
        count = base + (1 if i < extra else 0)
        subroute = nodes[index:index+count]
        index += count
        route = [start] + subroute
        routes.append(route)
    return routes

# -------------------------------
# 添加自定义起点到图中
# -------------------------------
def add_start_node(G, coords, start_coords=(50,50), start="start"):
    G.add_node(start)
    coords[start] = start_coords
    for node in list(G.nodes()):
        if node == start:
            continue
        x1, y1 = coords[node]
        x2, y2 = start_coords
        dist = math.hypot(x1-x2, y1-y2)
        G.add_edge(start, node, weight=dist)
    return G, coords

# -------------------------------
# Christofides 算法（单机器人开放解）
# -------------------------------
def christofides_tsp(graph, start="start"):
    # 1. 求最小生成树
    T = nx.minimum_spanning_tree(graph)
    # 2. 找出奇数度节点
    odd_degree_nodes = [v for v, d in T.degree() if d % 2 == 1]
    # 3. 在奇数节点上求最小权重完美匹配
    subgraph = nx.Graph()
    for i in range(len(odd_degree_nodes)):
        for j in range(i+1, len(odd_degree_nodes)):
            u = odd_degree_nodes[i]
            v = odd_degree_nodes[j]
            weight = graph[u][v]['weight']
            subgraph.add_edge(u, v, weight=weight)
    matching = nx.algorithms.matching.min_weight_matching(subgraph, weight='weight')
    # 4. 合并 MST 与匹配边
    multigraph = nx.MultiGraph()
    multigraph.add_edges_from(T.edges(data=True))
    for u, v in matching:
        multigraph.add_edge(u, v, weight=graph[u][v]['weight'])
    # 5. 求欧拉回路（闭合解）
    euler_circuit = list(nx.eulerian_circuit(multigraph, source=start))
    visited = set()
    closed_tour = []
    for u, v in euler_circuit:
        if u not in visited:
            closed_tour.append(u)
            visited.add(u)
    closed_tour.append(closed_tour[0])
    # 6. 去掉闭合环中权重最大的边，转换为开放解
    max_edge_weight = -1
    max_edge_index = -1
    for i in range(len(closed_tour) - 1):
        w = graph[closed_tour[i]][closed_tour[i+1]]['weight']
        if w > max_edge_weight:
            max_edge_weight = w
            max_edge_index = i
    open_tour = closed_tour[max_edge_index+1:-1] + closed_tour[0:max_edge_index+1]
    open_tour = ensure_start(open_tour, start)
    return open_tour

# -------------------------------
# ACO 算法（单机器人开放解），同时保存每次迭代的最佳解
# -------------------------------
def construct_tour(graph, nodes, pheromone, alpha, beta, start="start"):
    current = start
    tour = [current]
    unvisited = set(nodes)
    unvisited.remove(current)
    while unvisited:
        probs = []
        total = 0.0
        for next_node in unvisited:
            tau = pheromone.get((current, next_node), 1.0)
            eta = 1.0 / graph[current][next_node]['weight']
            prob = (tau ** alpha) * (eta ** beta)
            probs.append((next_node, prob))
            total += prob
        probs = [(node, prob/total) for node, prob in probs]
        r = random.random()
        cumulative = 0.0
        for node, p in probs:
            cumulative += p
            if r <= cumulative:
                next_node = node
                break
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return tour

def ant_colony_tsp(graph, init_tour, num_ants, num_iterations, alpha, beta, evaporation_rate, Q, start="start"):
    nodes = list(graph.nodes())
    pheromone = {}
    for u, v, data in graph.edges(data=True):
        pheromone[(u, v)] = 1.0
        pheromone[(v, u)] = 1.0

    best_tour = init_tour
    best_length = tour_length(graph, best_tour)
    best_history = [best_tour.copy()]  # 保存初始解
    for it in range(num_iterations):
        all_tours = []
        all_lengths = []
        for _ in range(num_ants):
            tour = construct_tour(graph, nodes, pheromone, alpha, beta, start)
            length = tour_length(graph, tour)
            all_tours.append(tour)
            all_lengths.append(length)
            if length < best_length:
                best_length = length
                best_tour = tour.copy()
        for key in pheromone:
            pheromone[key] *= (1 - evaporation_rate)
        for tour, length in zip(all_tours, all_lengths):
            deposit = Q / length
            for i in range(len(tour)-1):
                u = tour[i]
                v = tour[i+1]
                pheromone[(u, v)] += deposit
                pheromone[(v, u)] += deposit
        best_history.append(best_tour.copy())
        print(f"ACO Iteration {it+1}: Best tour length = {best_length:.2f}")
    return best_tour, best_length, best_history

def random_tour(graph, start="start"):
    nodes = list(graph.nodes())
    nodes.remove(start)
    random.shuffle(nodes)
    return [start] + nodes

# -------------------------------
# 模拟退火算法（单机器人开放解），保存每次迭代最佳解
# -------------------------------
def two_opt_swap(tour, i, k):
    new_tour = tour[:]
    new_tour[i:k+1] = new_tour[i:k+1][::-1]
    return new_tour

def simulated_annealing_tsp(graph, init_tour, initial_temp, cooling_rate, iterations, start="start"):
    current_tour = init_tour[:]
    current_length = tour_length(graph, current_tour)
    best_tour = current_tour[:]
    best_length = current_length
    best_history = [current_tour.copy()]
    temp = initial_temp
    for i in range(iterations):
        i_idx = random.randint(1, len(current_tour)-2)
        k_idx = random.randint(i_idx+1, len(current_tour)-1)
        new_tour = two_opt_swap(current_tour, i_idx, k_idx)
        new_length = tour_length(graph, new_tour)
        delta = new_length - current_length
        if delta < 0 or random.random() < math.exp(-delta/temp):
            current_tour = new_tour
            current_length = new_length
            if current_length < best_length:
                best_tour = current_tour.copy()
                best_length = current_length
        best_history.append(best_tour.copy())
        temp *= cooling_rate
    print(f"Simulated Annealing: Final tour length = {best_length:.2f}")
    return best_tour, best_length, best_history

# -------------------------------
# 遗传算法（单机器人开放解），保存每代最佳解
# -------------------------------
def order_crossover(parent1, parent2):
    size = len(parent1)
    child = [None] * size
    start_idx = random.randint(1, size-2)
    end_idx = random.randint(start_idx+1, size-1)
    child[start_idx:end_idx] = parent1[start_idx:end_idx]
    pos = end_idx
    for gene in parent2[1:]:
        if gene not in child:
            if pos >= size:
                pos = 1
            child[pos] = gene
            pos += 1
    child[0] = parent1[0]
    return child

def swap_mutation(tour, mutation_rate):
    new_tour = tour[:]
    for i in range(1, len(new_tour)):
        if random.random() < mutation_rate:
            j = random.randint(1, len(new_tour)-1)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour

def genetic_algorithm_tsp(graph, population_size, generations, mutation_rate, start="start"):
    nodes = list(graph.nodes())
    population = []
    for _ in range(population_size):
        other = nodes[:]
        other.remove(start)
        random.shuffle(other)
        tour = [start] + other
        population.append(tour)
    def fitness(tour):
        return 1.0 / tour_length(graph, tour)
    best_tour = min(population, key=lambda t: tour_length(graph, t))
    best_length = tour_length(graph, best_tour)
    best_history = [best_tour.copy()]
    for gen in range(generations):
        new_population = []
        while len(new_population) < population_size:
            tournament_size = 5
            tournament = random.sample(population, tournament_size)
            parent1 = max(tournament, key=lambda t: fitness(t))
            tournament = random.sample(population, tournament_size)
            parent2 = max(tournament, key=lambda t: fitness(t))
            child = order_crossover(parent1, parent2)
            child = swap_mutation(child, mutation_rate)
            new_population.append(child)
        population = new_population
        current_best = min(population, key=lambda t: tour_length(graph, t))
        current_length = tour_length(graph, current_best)
        if current_length < best_length:
            best_tour = current_best.copy()
            best_length = current_length
        best_history.append(best_tour.copy())
        print(f"GA Generation {gen+1}: Best tour length = {best_length:.2f}")
    print(f"Genetic Algorithm: Final tour length = {best_length:.2f}")
    return best_tour, best_length, best_history

# -------------------------------
# 随机图生成（完全图，边权为欧氏距离）
# -------------------------------
def create_random_graph(n, pattern="clusters", seed=42):
    random.seed(seed)
    np.random.seed(seed)
    coords = {}
    if pattern == "uniform":
        coords = {i: (np.random.rand()*100, np.random.rand()*100) for i in range(n)}
    elif pattern == "gaussian":
        coords = {i: (np.random.normal(50,15), np.random.normal(50,15)) for i in range(n)}
    elif pattern == "circle":
        coords = {}
        for i in range(n):
            angle = 2*math.pi*i/n
            r = 40
            x = 50 + r*math.cos(angle)
            y = 50 + r*math.sin(angle)
            coords[i] = (x,y)
    elif pattern == "clusters":
        num_clusters = 3
        cluster_centers = [(np.random.rand()*100, np.random.rand()*100) for _ in range(num_clusters)]
        coords = {}
        cluster_size = n // num_clusters
        idx = 0
        for c in range(num_clusters):
            cx, cy = cluster_centers[c]
            for _ in range(cluster_size):
                x = np.random.normal(cx, 5)
                y = np.random.normal(cy, 5)
                coords[idx] = (x,y)
                idx += 1
        while idx < n:
            cx, cy = random.choice(cluster_centers)
            x = np.random.normal(cx, 5)
            y = np.random.normal(cy, 5)
            coords[idx] = (x,y)
            idx += 1
    else:
        raise ValueError("Unknown pattern.")
    G = nx.complete_graph(n)
    for u, v in G.edges():
        x1, y1 = coords[u]
        x2, y2 = coords[v]
        dist = math.hypot(x1-x2, y1-y2)
        G[u][v]['weight'] = dist
    return G, coords

# -------------------------------
# 绘制多机器人路径（折线图），用星号标记起点
# -------------------------------
def plot_multi_robot_routes(coords, routes, title="Multi-Robot Routes", save_path=None):
    plt.figure(figsize=(8,6))
    colors = ['r', 'g', 'b', 'orange', 'purple', 'cyan']
    for i, route in enumerate(routes):
        x = [coords[node][0] for node in route]
        y = [coords[node][1] for node in route]
        plt.plot(x, y, 'o-', color=colors[i % len(colors)], label=f"Robot {i+1}")
        plt.plot(x[0], y[0], marker="*", markersize=15, color=colors[i % len(colors)])
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# -------------------------------
# 动画函数：展示每次迭代时的多机器人路径
# -------------------------------
def animate_routes_history(coords, tours_history, num_robots, title="Route Evolution", start="start"):
    fig, ax = plt.subplots(figsize=(8,6))
    colors = ['r', 'g', 'b', 'orange', 'purple', 'cyan']
    lines = []
    stars = []
    for i in range(num_robots):
        line, = ax.plot([], [], 'o-', color=colors[i % len(colors)], label=f"Robot {i+1}")
        lines.append(line)
        star, = ax.plot([], [], marker="*", markersize=15, color=colors[i % len(colors)])
        stars.append(star)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)
    
    xs = [coord[0] for coord in coords.values()]
    ys = [coord[1] for coord in coords.values()]
    ax.set_xlim(min(xs)-5, max(xs)+5)
    ax.set_ylim(min(ys)-5, max(ys)+5)
    
    def update(frame):
        tour = tours_history[frame]
        routes = split_tour_among_robots(tour, num_robots, start)
        for i, route in enumerate(routes):
            x = [coords[node][0] for node in route]
            y = [coords[node][1] for node in route]
            lines[i].set_data(x, y)
            stars[i].set_data(x[0], y[0])
        ax.set_title(f"{title} - Iteration {frame}")
        return lines + stars
    
    ani = animation.FuncAnimation(fig, update, frames=len(tours_history),
                                  interval=500, blit=True, repeat=False)
    plt.show()

# -------------------------------
# 新增功能：记录每次迭代中多机器人路径及路径长度
# -------------------------------
def record_history(tours_history, num_robots, G, start):
    """
    对每次迭代的总体巡游（tours_history）进行拆分，记录：
      - 每次迭代的多机器人路径 (multi_history)
      - 每次迭代各机器人路径长度 (robot_history, list of lists)
      - 每次迭代总体路径长度（各机器人路径长度之和）
      - 每次迭代最大单机器人路径长度
    """
    multi_history = []
    total_history = []
    max_history = []
    robot_history = [[] for _ in range(num_robots)]
    for tour in tours_history:
        routes = split_tour_among_robots(tour, num_robots, start)
        multi_history.append(routes)
        tot = sum(tour_length(G, route) for route in routes)
        total_history.append(tot)
        maximum = max(tour_length(G, route) for route in routes)
        max_history.append(maximum)
        for i, route in enumerate(routes):
            robot_history[i].append(tour_length(G, route))
    return multi_history, total_history, max_history, robot_history

# -------------------------------
# 新增功能：折线图显示各算法中每次迭代的各机器人路径长度变化及总体路径长度
# -------------------------------
def plot_history_line_chart(robot_history, total_history, algorithm_name):
    iterations = range(len(total_history))
    plt.figure(figsize=(10,6))
    for i, robot_lengths in enumerate(robot_history):
        plt.plot(iterations, robot_lengths, marker='o', label=f"Robot {i+1} 路径长度")
    plt.plot(iterations, total_history, marker='x', linestyle='--', label="总体路径长度")
    plt.xlabel("迭代次数")
    plt.ylabel("路径长度")
    plt.title(f"{algorithm_name} - 路径长度演化")
    plt.legend()
    plt.grid(True)
    plt.show()
