import numpy as np
import matplotlib.pyplot as plt
import time
import os
import tsplib95

def calc_distance_matrix(cities):
    n = len(cities)
    dist_matrix = np.sqrt(((cities[np.newaxis, :, :] - cities[:, np.newaxis, :]) ** 2).sum(axis=2))
    dist_matrix += np.eye(n) * 1e-9  
    return dist_matrix

def plot(cities, tour, distance_matrix, title=""):
    plt.figure(figsize=(10, 7))
    plt.title(title)
    n = len(cities)
    for i in range(-1, n - 1):
        plt.plot([cities[tour[i], 0], cities[tour[i + 1], 0]], [cities[tour[i], 1], cities[tour[i + 1], 1]], 'b-')
        dist = distance_matrix[tour[i], tour[i+1]]
        mid_point = (cities[tour[i], 0] + cities[tour[i+1], 0])/2, (cities[tour[i], 1] + cities[tour[i+1], 1])/2
        plt.text(mid_point[0], mid_point[1], f"{dist:.2f}", color='purple', fontsize=8)
    plt.scatter(cities[:, 0], cities[:, 1], c='red')
    for i, city in enumerate(cities):
        plt.text(city[0], city[1], str(i))
    plt.show()

def aco_2opt(cities, distance_matrix, alpha, beta, n_ants=20, n_iterations=100, rho=0.5):
    n = len(cities)
    pheromone = np.ones((n, n)) / n
    best_tour = None
    best_distance = np.inf
   
    for iteration in range(n_iterations):
        all_tours = []
        for ant in range(n_ants):
            tour = construct_tour(n, distance_matrix, pheromone, alpha, beta)
            tour = two_opt(tour, distance_matrix)  
            tour_distance = calc_tour_distance(tour, distance_matrix)
            all_tours.append((tour, tour_distance))
            if tour_distance < best_distance:
                best_tour = tour[:]
                best_distance = tour_distance
       
        pheromone *= (1 - rho)  
        for tour, tour_distance in all_tours:
            for i in range(len(tour) - 1):
                pheromone[tour[i], tour[i + 1]] += 1.0 / tour_distance  
       
    return best_tour

def construct_tour(n, distance_matrix, pheromone, alpha, beta):
    tour = [np.random.randint(n)]
    visited = set(tour)
    while len(visited) < n:
        probabilities = ((pheromone[tour[-1]] ** alpha) * ((1.0 / distance_matrix[tour[-1]]) ** beta))
        probabilities[list(visited)] = 0  
        probabilities /= probabilities.sum()
        next_city = np.random.choice(np.arange(n), p=probabilities)
        tour.append(next_city)
        visited.add(next_city)
    return tour

def two_opt(tour, distance_matrix):
    n = len(tour)
    best_distance = calc_tour_distance(tour, distance_matrix)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue  
                new_tour = tour[:]
                new_tour[i:j] = reversed(tour[i:j])  
                new_distance = calc_tour_distance(new_tour, distance_matrix)
                if new_distance < best_distance:
                    tour = new_tour
                    best_distance = new_distance
                    improved = True
        if improved:
            break
    return tour

def calc_tour_distance(tour, distance_matrix):
    return np.sum(distance_matrix[np.array(tour), np.roll(np.array(tour), -1)])

def read_tsp(file_path):
    tsp = tsplib95.load(file_path)
    coordinates = np.array([tsp.node_coords[i+1] for i in range(tsp.dimension)])
    return coordinates

def ant_algorithm(file_path, optimal_distance=None):
    coordinates = read_tsp(file_path)
    instance_name = os.path.basename(file_path)
    cities = np.array(coordinates)
    dist_matrix = calc_distance_matrix(cities)
   
    alpha = 1
    beta = 2
   
    start_time = time.time()
    tour_aco = aco_2opt(cities, dist_matrix, alpha, beta)
    aco_duration = time.time() - start_time
    aco_distance = calc_tour_distance(tour_aco, dist_matrix)
    print("Instance Name:", instance_name)
    print("Optimal Path:", tour_aco)
    print("Time Taken:", aco_duration, "seconds")
    print("ACO Distance:", aco_distance)
    
    if optimal_distance is not None:
        error_percentage = ((aco_distance - optimal_distance) / optimal_distance) * 100
        print("Optimal Distance:", optimal_distance)
        print("Error Percentage:", error_percentage, "%")
    else:
        print("Optimal Distance: Not Provided")
    
    print("Distances between consecutive points:")
    total_distance = 0
    for i in range(len(tour_aco)):
        city1 = tour_aco[i]
        city2 = tour_aco[(i + 1) % len(tour_aco)]  
        distance = dist_matrix[city1, city2]
        total_distance += distance
        print(f"Distance between points {city1} and {city2}: {distance}")
    
    plot(cities, tour_aco, dist_matrix, f"Optimal Path using ACO with 2-opt for {instance_name}")

file_path = r'C:\Users\Bhuvana\Downloads\eil51.tsp'
ant_algorithm(file_path, 426)
