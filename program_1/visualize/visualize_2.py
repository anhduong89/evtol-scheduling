import networkx as nx
import matplotlib.pyplot as plt
import os
import re
import numpy as np

city = ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
dist = np.array(
    [
        [0., 10.48113508, 18.51467981, 18.54127528, 5.7385456, 13.03650962, 20.09902115], 
        [10.48113508, 0., 9.15619145, 15.04929904, 10.56004164, 6.52505341, 13.22524436],
        [18.51467981, 9.15619145, 0., 11.47944457, 16.1038714, 6.31668958, 6.33708216],
        [18.54127528, 15.04929904, 11.47944457, 0., 13.32823709, 8.5872692, 6.16996146],
        [5.7385456, 10.56004164, 16.1038714, 13.32823709, 0., 9.86908414, 16.01052999],
        [13.03650962, 6.52505341, 6.31668958, 8.5872692, 9.86908414, 0., 7.31033037],
        [20.09902115, 13.22524436, 6.33708216, 6.16996146, 16.01052999, 7.31033037, 0.]
    ]
)

def parse_flight_data(line):
    pattern = r'as\((\d+),\((\w+),(\w+)\),(\d+),(\d+)\)'
    match = re.match(pattern, line.strip())
    if match:
        return int(match.group(1)), (match.group(2), match.group(3)), int(match.group(4)), int(match.group(5))
    return None

def parse_gate_assign_data(line):
    pattern = r'as_gate\(\((\d+),(\d+)\),\((\d+),(\w+)\)\)'
    match = re.match(pattern, line.strip())
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), match.group(4)
    return None

def parse_time_data(line):
    start_pattern = r'dl\(start\((\d+),\((\w+),(\w+)\),(\d+)\),(\d+)\)'
    arrival_pattern = r'dl\(arrival\((\d+),\((\w+),(\w+)\),(\d+)\),(\d+)\)'
    start_match = re.match(start_pattern, line.strip())
    arrival_match = re.match(arrival_pattern, line.strip())
    if start_match:
        return 'start', int(start_match.group(1)), int(start_match.group(4)), int(start_match.group(5))
    if arrival_match:
        return 'arrival', int(arrival_match.group(1)), int(arrival_match.group(4)), int(arrival_match.group(5))
    return None

def visualize():
    start_times, arrival_times, flights, gates = [], [], [], []
    time_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'trajectories.lp')
    try:
        with open(time_data_path, 'r') as file:
            for data in file:
                for line in data.strip().split(" "):
                    time_data = parse_time_data(line)
                    flight_data = parse_flight_data(line)
                    gate_data = parse_gate_assign_data(line)
                    if time_data:
                        data_type, agent_id, segment_id, time = time_data
                        while len(start_times) <= agent_id:
                            start_times.append([])
                            arrival_times.append([])
                        while len(start_times[agent_id]) <= segment_id:
                            start_times[agent_id].append(None)
                            arrival_times[agent_id].append(None)
                        if data_type == 'start':
                            start_times[agent_id][segment_id] = time
                        elif data_type == 'arrival':
                            arrival_times[agent_id][segment_id] = time
                    if flight_data:
                        flights.append(flight_data)
                    if gate_data:
                        gates.append(gate_data)
    except FileNotFoundError:
        print(f"Error: Could not find file {time_data_path}")
        exit(1)

    agent_flights = {}
    for agent, (origin, destination), passengers, segment in flights:
        agent_flights.setdefault(agent, []).append((segment, origin, destination, passengers))

    agent_gates = {}
    for agent_id, segment, gate_id, destination in gates:
        agent_gates.setdefault(agent_id, {}).setdefault(segment, []).append((gate_id, destination))

    output_dir = os.path.dirname(__file__)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    G = nx.Graph()
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    plt.figure(figsize=(15, 8))

    pos = {}
    y_spacing = 1.0
    add = [i * 4 for i in range(len(arrival_times[0]))]
    arrival_times_true = arrival_times
    arrival_times = [[x + y if x is not None else None for x, y in zip(a, add)] for a in arrival_times]
    start_times_true = start_times
    start_times = [[x + y if x is not None else None for x, y in zip(a, add)] for a in start_times]

    for agent, segments in agent_flights.items():
        y_pos = -agent * y_spacing

        # Add agent label at the beginning of the graph
        agent_label_node = f"Agent_{agent}_label"
        pos[agent_label_node] = (-2, y_pos)  # Position the label slightly to the left
        G.add_node(agent_label_node)
        G.nodes[agent_label_node]['label'] = f"Agent {agent}:"

        for seg_num, origin, destination, passengers in sorted(segments, key=lambda x: x[0]):
            src_node = f"{origin}_{agent}_{seg_num}"
            depart_node = f"{origin}_{agent}_{seg_num}_depart"
            dst_node = f"{destination}_{agent}_{seg_num}"

            if seg_num == 1:
                pos[src_node] = (0, y_pos)
                pos[depart_node] = (0, y_pos)
                pos[dst_node] = (arrival_times[agent][seg_num] / 5, y_pos)
            else:
                pos[src_node] = (arrival_times[agent][seg_num - 1] / 5, y_pos)
                pos[depart_node] = (start_times[agent][seg_num] / 5, y_pos)
                pos[dst_node] = (arrival_times[agent][seg_num] / 5, y_pos)

            G.add_node(src_node)
            G.add_node(depart_node)
            G.add_node(dst_node)
            G.add_edge(src_node, depart_node)
            nx.draw_networkx_edges(G, pos, edgelist=[(src_node, depart_node)], style='dotted', edge_color='black', width=0.5)
            G.add_edge(depart_node, dst_node)
            edge_color = 'gray' if passengers == 0 else 'red' if passengers == 4 else 'black'
            nx.draw_networkx_edges(G, pos, edgelist=[(depart_node, dst_node)], style='solid', edge_color=edge_color, width=0.5)

            # Add edge label for depart_node to dst_node under the edge
            edge_label = f"S{seg_num}\nP={passengers}"
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels={(depart_node, dst_node): edge_label}, font_size=7, font_color='black', label_pos=0.5, bbox=dict(alpha=0)
            )

            G.nodes[src_node]['label'] = origin if seg_num == 1 else ""
            G.nodes[depart_node]['label'] = origin
            G.nodes[dst_node]['label'] = destination

            gate_info = agent_gates.get(agent, {}).get(seg_num, [])
            if gate_info:
                gate_id, _ = gate_info[0]
                G.nodes[dst_node]['label'] += f":{gate_id}"

            gate_info_depart = agent_gates.get(agent, {}).get(seg_num - 1, [])
            if gate_info_depart:
                gate_id_depart, _ = gate_info_depart[0]
                G.nodes[depart_node]['label'] += f":{gate_id_depart}"

            if start_times[agent][seg_num] is not None:
                G.nodes[depart_node]['label'] = f"\n{start_times_true[agent][seg_num]}\u2192\n" + G.nodes[depart_node]['label']
            if arrival_times[agent][seg_num] is not None:
                G.nodes[dst_node]['label'] = f"\n{arrival_times_true[agent][seg_num]}\n" + G.nodes[dst_node]['label']

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=5, node_color='lightblue')
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=11, verticalalignment='bottom')

    plt.title("Sequential Flight Segments for All Agents")
    plt.axis('off')
    plt.tight_layout()

    output_path = os.path.join(output_dir, "all_agents_trajectory.svg")
    plt.savefig(output_path, dpi=1000, format="svg", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    visualize()