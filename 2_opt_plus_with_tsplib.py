import os
import numpy as np
import time
import matplotlib.pyplot as plt
import tsplib95

def calculate_distance(coords):
    n = len(coords)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distance = np.linalg.norm(coords[i] - coords[j])
            distances[i, j] = distances[j, i] = distance
    return distances

def greedy_tour(distances):
    n = distances.shape[0]
    tour = [0]
    unvisited = set(range(1, n))
    while unvisited:
        last = tour[-1]
        next_city = min(unvisited, key=lambda city: distances[last, city])
        unvisited.remove(next_city)
        tour.append(next_city)
    return tour

def calculate_tour_length(tour, distances):
    return sum(distances[tour[i], tour[i+1]] for i in range(len(tour)-1)) + distances[tour[-1], tour[0]]

def two_opt_swap(tour, i, k):
    new_tour = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
    return new_tour

def two_opt(tour, distances):
    improvement = True
    while improvement:
        improvement = False
        for i in range(1, len(tour) - 2):
            for k in range(i+1, len(tour)):
                new_tour = two_opt_swap(tour, i, k)
                if calculate_tour_length(new_tour, distances) < calculate_tour_length(tour, distances):
                    tour = new_tour
                    improvement = True
                    break
            if improvement:
                break
    return tour

def two_opt_plus(coords, instance_name, optimal_distance=None, num_iterations=100):
    start_time = time.time()
    
    n = len(coords)
    distances = calculate_distance(coords)
    tour = greedy_tour(distances)

    for _ in range(num_iterations):
        tour = two_opt(tour, distances)

    tour_length = calculate_tour_length(tour, distances)
    print("Instance Name:", instance_name)
    print("Number of Iterations:", num_iterations)
    print("Tour Length:", tour_length)

    if optimal_distance is not None:
        error_percentage = ((tour_length - optimal_distance) / optimal_distance) * 100
        print("Optimal Distance:", optimal_distance)
        print(f"Error Percentage: {error_percentage:.2f}%")

    time_taken = time.time() - start_time
    print("Total Time:", time_taken, "seconds")
    print("Distances between consecutive points:")
    total_distance = 0
    for i in range(len(tour)):
        city1 = tour[i]
        city2 = tour[(i + 1) % len(tour)]  
        distance = distances[city1, city2]
        total_distance += distance
        print(f"Distance between points {city1} and {city2}: {distance:.2f}")

    plot_tour(coords, tour, distances, instance_name)

def read_tsp_file(file_path):
    tsp = tsplib95.load(file_path)
    coordinates = np.array([tsp.node_coords[i+1] for i in range(tsp.dimension)])
    return coordinates

def plot_tour(coords, tour, distances, instance_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c='blue', marker='o')
    total_distance = 0
    for i in range(len(tour)):
        start, end = tour[i], tour[(i + 1) % len(tour)]
        plt.plot([coords[start][0], coords[end][0]], [coords[start][1], coords[end][1]], 'r-')
        plt.text(coords[start][0], coords[start][1], f'{start}', fontsize=9)
        distance = distances[start, end]
        total_distance += distance
        mid_x, mid_y = (coords[start][0] + coords[end][0]) / 2, (coords[start][1] + coords[end][1]) / 2
        plt.text(mid_x, mid_y, f'{distance:.2f}', color='purple', fontsize=8)

    plt.title(f"Optimal Path using 2-opt++ for {instance_name}")
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.show()

file_path = r'C:\Users\Bhuvana\Downloads\eil51.tsp' #path of the dataset file
instance_name = os.path.basename(file_path)
coordinates = read_tsp_file(file_path)
two_opt_plus(coordinates, instance_name, 426)
