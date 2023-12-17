import tkinter as tk
from tkinter import ttk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import time
import random
import math

class Node:
    def __init__(self, name, neighbors):
        self.name = name
        self.neighbors = neighbors

def generate_neighbors(graph, node):
    # Retrieve neighbors from the graph
    if node in graph:
        return graph[node].neighbors
    else:
        return []

class PathfindingSimulator:
    def __init__(self, master):
        self.master = master
        self.master.title("Pathfinding Simulator")

        self.canvas_frame = tk.Frame(self.master)
        self.canvas_frame.pack()

        self.algorithm_var = tk.StringVar()
        self.algorithm_var.set("BFS")

        self.algorithm_menu = ttk.Combobox(self.master, textvariable=self.algorithm_var, values=["BFS", "DFS", "Hill Climbing", "Beam", "Branch and Bound", "A*"])
        self.algorithm_menu.pack(pady=10)

        self.start_node = tk.Label(self.master, text="Start")
        self.start_node.pack(pady=5)

        self.start_node_entry = tk.Entry(self.master, textvariable=tk.StringVar(), justify="center")
        self.start_node_entry.pack(pady=5)

        self.goal_node = tk.Label(self.master, text="Goal")
        self.goal_node.pack(pady=5)

        self.goal_node_entry = tk.Entry(self.master, textvariable=tk.StringVar(), justify="center")
        self.goal_node_entry.pack(pady=5)

        self.play_button = tk.Button(self.master, text="Play", command=self.start_simulation)
        self.play_button.pack(side=tk.LEFT)

        self.pause_button = tk.Button(self.master, text="Pause", command=self.pause_simulation, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT)

        self.start_simulation_button = tk.Button(self.master, text="Start Simulation", command=self.reset_and_start_simulation)
        self.start_simulation_button.pack(side=tk.LEFT)

        self.speed_scale = tk.Scale(self.master, label="Simulation Speed", from_=1, to=10, orient=tk.HORIZONTAL)
        self.speed_scale.set(5)
        self.speed_scale.pack()

        self.enqueue_label = tk.Label(self.master, text="Enqueues: 0")
        self.enqueue_label.pack()

        self.queue_size_label = tk.Label(self.master, text="Queue Size: 0")
        self.queue_size_label.pack()

        self.path_elements_label = tk.Label(self.master, text="Path Elements: ")
        self.path_elements_label.pack()

        self.path_found_label = tk.Label(self.master, text="Path Found: ")
        self.path_found_label.pack()

        self.graph = {
            'A': Node('A', ['B', 'C']),
            'B': Node('B', ['A', 'D', 'E']),
            'C': Node('C', ['A', 'F', 'G']),
            'D': Node('D', ['B']),
            'E': Node('E', ['B', 'H']),
            'F': Node('F', ['C']),
            'G': Node('G', ['C']),
            'H': Node('H', ['E'])
        }

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack()

        self.pos = nx.spring_layout(self.create_graph())
        self.initial_graph = self.create_graph()

        self.draw_graph()

    def start_simulation(self):
        start_node = self.start_node_entry.get().upper()  # Convert to uppercase for consistency
        goal_node = self.goal_node_entry.get().upper()

        if start_node not in self.graph or goal_node not in self.graph:
            print("Invalid start or goal node. Please enter valid nodes.")
            return

        algorithm = self.algorithm_var.get()

        if algorithm not in ["BFS", "DFS"]:
            self.graph = self.create_graph()

        self.initialize_simulation(start_node)

        self.pause_flag = False

        if algorithm == "BFS":
            self.simulate_bfs(goal_node)
        elif algorithm == "DFS":
            self.simulate_dfs(goal_node)
        elif algorithm == "Hill Climbing":
            self.simulate_hill_climbing(goal_node)
        else:
            self.simulate_weighted_algorithm(algorithm, goal_node)

    def simulate_bfs(self, goal_node):
        simulation_speed = self.speed_scale.get()
        enqueue_count = 0

        self.play_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.start_simulation_button.config(state=tk.DISABLED)

        while self.queue and not self.pause_flag:
            time.sleep(1 / simulation_speed)

            current_node, path = self.queue.popleft()
            if current_node == goal_node:
                print("Path found:", path)
                self.path_found_label.config(text=f"Path Found: {path}")
                break

            if current_node not in self.visited:
                self.visited.add(current_node)

                if current_node in self.graph:
                    extensions = self.graph[current_node].neighbors
                    enqueue_count += len(extensions) 
                    for neighbor in extensions:
                        if neighbor not in self.visited:
                            new_path = list(path)
                            new_path.append(neighbor)
                            self.queue.append((neighbor, new_path))

                self.update_path(path, current_node, extensions)
                self.enqueue_label.config(text=f"Enqueues: {enqueue_count}")
                self.queue_size_label.config(text=f"Queue Size: {len(self.queue)}")
                self.path_elements_label.config(text=f"Path Elements: {path}")

                print("Algorithm: BFS")
                print("Iteration:", len(self.visited))
                print("Enqueues:", enqueue_count)
                print("Queue Size:", len(self.queue))
                print("Path Elements:", path)

                self.master.update()

        self.play_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.start_simulation_button.config(state=tk.NORMAL)
        self.pause_flag = False

    def simulate_dfs(self, goal_node):
        simulation_speed = self.speed_scale.get()
        enqueue_count = 0

        self.play_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.start_simulation_button.config(state=tk.DISABLED)

        while self.queue and not self.pause_flag:
            time.sleep(1 / simulation_speed)

            current_node, path = self.queue.pop()
            if current_node == goal_node:
                print("Path found:", path)
                self.path_found_label.config(text=f"Path Found: {path}")
                break

            if current_node not in self.visited:
                self.visited.add(current_node)

                if current_node in self.graph:
                    extensions = self.graph[current_node].neighbors
                    enqueue_count += len(extensions)
                    for neighbor in extensions:
                        if neighbor not in self.visited:
                            new_path = list(path)
                            new_path.append(neighbor)
                            self.queue.append((neighbor, new_path))

                self.update_path(path, current_node, extensions)
                self.enqueue_label.config(text=f"Enqueues: {enqueue_count}")
                self.queue_size_label.config(text=f"Queue Size: {len(self.queue)}")
                self.path_elements_label.config(text=f"Path Elements: {path}")

                print("Algorithm: DFS")
                print("Iteration:", len(self.visited))
                print("Enqueues:", enqueue_count)
                print("Queue Size:", len(self.queue))
                print("Path Elements:", path)

                self.master.update()

        self.play_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.start_simulation_button.config(state=tk.NORMAL)
        self.pause_flag = False

    def simulate_hill_climbing(self, goal_node):
        simulation_speed = self.speed_scale.get()
        enqueue_count = 0

        self.play_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.start_simulation_button.config(state=tk.DISABLED)

        start_node = self.start_node_entry.get().upper()  # Convert to uppercase for consistency
        current_node = start_node
        path = [start_node]

        while current_node != goal_node and not self.pause_flag:
            time.sleep(1 / simulation_speed)

            neighbors = generate_neighbors(self.graph, current_node)
            best_neighbor = max(neighbors, key=self.euclidean_distance)

            if self.euclidean_distance(best_neighbor, goal_node) >= self.euclidean_distance(current_node, goal_node):
                print("No better neighbor found. Stopping.")
                break

            current_node = best_neighbor
            path.append(current_node)

            self.update_path(path, current_node, neighbors)
            enqueue_count += 1

            self.enqueue_label.config(text=f"Enqueues: {enqueue_count}")
            self.queue_size_label.config(text=f"Queue Size: N/A")  # Not applicable for hill climbing
            self.path_elements_label.config(text=f"Path Elements: {path}")

            print("Algorithm: Hill Climbing")
            print("Iteration:", len(path) - 1)
            print("Enqueues:", enqueue_count)
            print("Queue Size: N/A")
            print("Path Elements:", path)

            self.master.update()

        if current_node == goal_node:
            print("Path found:", path)
            self.path_found_label.config(text=f"Path Found: {path}")

        self.play_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.start_simulation_button.config(state=tk.NORMAL)
        self.pause_flag = False

    def simulate_weighted_algorithm(self, algorithm, goal_node):
        print(f"Simulation of {algorithm} is not implemented yet.")

    def euclidean_distance(self, node, goal_node):
        x1, y1 = self.pos[node]
        x2, y2 = self.pos[goal_node]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def pause_simulation(self):
        self.pause_flag = True

    def reset_and_start_simulation(self):
        self.reset_simulation()
        self.draw_graph()
        self.start_simulation()

    def reset_simulation(self):
        self.visited = set()
        self.queue = deque()

    def create_graph(self):
        G = nx.Graph()

        for node, data in self.graph.items():
            G.add_node(node)
            if self.algorithm_var.get() not in ["BFS", "DFS"]:
                for neighbor in data.neighbors:
                    weight = random.randint(1, 10)
                    G.add_edge(node, neighbor, weight=weight)
            else:
                for neighbor in data.neighbors:
                    G.add_edge(node, neighbor)

        return G

    def draw_graph(self):
        edge_labels = {(edge[0], edge[1]): data['weight'] for edge, data in self.initial_graph.edges.items() if 'weight' in data}
        nx.draw(self.initial_graph, self.pos, with_labels=True, ax=self.ax, node_color='skyblue', node_size=700,
                font_size=10, font_color='black', font_weight='bold')
        nx.draw_networkx_edge_labels(self.initial_graph, self.pos, edge_labels=edge_labels, font_color='red')
        self.canvas.draw()

    def update_path(self, path, current_node, extensions):
        nx.draw_networkx_edges(self.initial_graph, self.pos, edgelist=[(path[i], path[i + 1]) for i in range(len(path) - 1)],
                               edge_color='red', width=2)

        nx.draw_networkx_nodes(self.initial_graph, self.pos, nodelist=[current_node], node_color='green', node_size=700)

        self.canvas.draw()

    def initialize_simulation(self, start_node):
        self.visited = set()
        self.queue = deque()
        self.queue.append((start_node, [start_node]))
        self.draw_graph()


    def generate_new_graph(self):
        if self.algorithm_var.get() not in ["BFS", "DFS"]:
        # If the selected algorithm is not BFS or DFS, create a new graph
            self.graph = self.create_graph()
        else:
        # Otherwise, use the default graph structure
            self.graph = {
                'A': Node('A', ['B', 'C']),
                'B': Node('B', ['A', 'D', 'E']),
                'C': Node('C', ['A', 'F', 'G']),
                'D': Node('D', ['B']),
                'E': Node('E', ['B', 'H']),
                'F': Node('F', ['C']),
                'G': Node('G', ['C']),
                'H': Node('H', ['E'])
            }

        # Clear the graph, reset simulation, and create a new layout
            self.graph.clear()
            self.reset_simulation()
            self.pos = nx.spring_layout(self.graph)

        # Create the initial graph structure
            self.initial_graph = self.create_graph()

        # Draw the graph
            self.draw_graph()

    
    def algorithm_changed(self, event):
        self.algorithm_menu.bind("<<ComboboxSelected>>", self.algorithm_changed)
        self.generate_new_graph()

def main():
    root = tk.Tk()
    app = PathfindingSimulator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
