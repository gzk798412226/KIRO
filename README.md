# KIRO - Multi-Robot Path Planning System

A comprehensive system for solving multi-robot path planning problems using various optimization algorithms.

## Overview

This project implements and compares multiple algorithms for solving the Traveling Salesman Problem (TSP) and its multi-robot variant. The system includes both classical and enhanced versions of popular optimization algorithms.

## Implemented Algorithms

1. **MST+ACO (Christofides + Ant Colony Optimization)**
   - Combines Christofides algorithm with ACO
   - Parameters: num_ants, num_iterations, alpha, beta, evaporation_rate, Q

2. **Pure ACO (Ant Colony Optimization)**
   - Standard implementation of ACO
   - Parameters: num_ants, num_iterations, alpha, beta, evaporation_rate, Q

3. **Simulated Annealing (SA)**
   - Classical simulated annealing implementation
   - Parameters: initial_temp, cooling_rate, iterations

4. **Genetic Algorithm (GA)**
   - Standard genetic algorithm implementation
   - Parameters: population_size, generations, mutation_rate

5. **AR-ACO (Adaptive Radius ACO)**
   - Enhanced ACO with adaptive radius
   - Parameters: num_ants, num_iterations, alpha, beta, evaporation_rate, Q, k

6. **IEACO (Improved Elite ACO)**
   - ACO with improved elite strategy
   - Parameters: num_ants, num_iterations, alpha_init, beta_init, evaporation_rate, Q, epsilon

7. **DL-ACO (Double Layer ACO)**
   - Two-layer ACO implementation
   - Parameters: num_ants_layer1, iters_layer1, num_ants_layer2, iters_layer2, alpha, beta, evaporation_rate, Q

8. **Smooth ACO**
   - ACO with smoothness constraints
   - Parameters: num_ants, num_iterations, alpha, beta, evaporation_rate, Q, angle_penalty

## Features

- Multi-robot path planning
- Random graph generation with uniform distribution
- Path visualization for each algorithm
- Comparative analysis of algorithm performance
- Support for custom start positions
- Configurable number of robots

## Usage

The main script (`main.py`) demonstrates the usage of all algorithms. Key parameters include:

```python
num_robots = 4  # Number of robots
start_coords = (50, 50)  # Starting coordinates
start_node = "start"  # Starting node identifier
```

## Visualization

The system provides:
- Static path visualization for each algorithm
- Comparative bar charts showing:
  - Total path length
  - Maximum single-robot path length

## Requirements

- Python 3.x
- NetworkX
- Matplotlib
- NumPy

## License

[Add your license information here]