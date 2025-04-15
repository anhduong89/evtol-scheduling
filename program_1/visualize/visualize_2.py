import networkx as nx
import matplotlib.pyplot as plt
import os
import re
import numpy as np
city = ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
dist = np.array(
    [
        [0.,          10.48113508,  18.51467981,  18.54127528,  5.7385456,    13.03650962,  20.09902115], 
        [10.48113508,  0.,           9.15619145,   15.04929904,  10.56004164,  6.52505341,   13.22524436],
        [18.51467981,  9.15619145,   0.,           11.47944457,  16.1038714,   6.31668958,   6.33708216 ],
        [18.54127528,  15.04929904,  11.47944457,  0.,           13.32823709,  8.5872692,    6.16996146 ],
        [5.7385456,    10.56004164,  16.1038714,   13.32823709,  0.,           9.86908414,   16.01052999],
        [13.03650962,  6.52505341,   6.31668958,   8.5872692,    9.86908414,   0.,           7.31033037 ],
        [20.09902115,  13.22524436,  6.33708216,   6.16996146,   16.01052999,  7.31033037,   0.         ]
        ]
    )

def parse_flight_data(line):
    # Pattern to match: as(<agent_ID>,(<origin>, <destination>), <passengers>, <segment>)
    pattern = r'as\((\d+),\((\w+),(\w+)\),(\d+),(\d+)\)'
    match = re.match(pattern, line.strip())
    if match:
        agent_id = int(match.group(1))
        origin = match.group(2)
        destination = match.group(3)
        passengers = int(match.group(4))
        segment = int(match.group(5))
        return (agent_id, (origin, destination), passengers, segment)
    return None

def parse_gate_assign_data(line):
    # Pattern to match: as_gate((<agent_ID>,<segment>),(<gate_ID>,<destination>))
    pattern = r'as_gate\(\((\d+),(\d+)\),\((\d+),(\w+)\)\)'
    match = re.match(pattern, line.strip())
    if match:
        agent_id = int(match.group(1))
        segment = int(match.group(2))
        gate_id = int(match.group(3))
        destination = match.group(4)
        return (agent_id, segment, gate_id, destination)
    return None
    

def parse_time_data(line):
    # Pattern to match: dl(start(<agent_ID>,(<origin>,<destination>),<segment_ID>),<time>)
    start_pattern = r'dl\(start\((\d+),\((\w+),(\w+)\),(\d+)\),(\d+)\)'
    arrival_pattern = r'dl\(arrival\((\d+),\((\w+),(\w+)\),(\d+)\),(\d+)\)'
    
    start_match = re.match(start_pattern, line.strip())
    arrival_match = re.match(arrival_pattern, line.strip())
    
    if start_match:
        agent_id = int(start_match.group(1))
        origin = start_match.group(2)
        destination = start_match.group(3)
        segment_id = int(start_match.group(4))
        time = int(start_match.group(5))
        return ('start', agent_id, segment_id, time)
    
    if arrival_match:
        agent_id = int(arrival_match.group(1))
        origin = arrival_match.group(2)
        destination = arrival_match.group(3)
        segment_id = int(arrival_match.group(4))
        time = int(arrival_match.group(5))
        return ('arrival', agent_id, segment_id, time)
    
    return None

def visualize():
    # Initialize lists for start and arrival times
    start_times = []
    arrival_times = []
    flights = []
    gates =[]
    # Read time data from file
    time_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'trajectories.lp')
    try:
        with open(time_data_path, 'r') as file:
            for line_number, data in enumerate(file):
                # if line_number == 4:  # Read only the 5th line (0-indexed)
                split_line = data.strip().split(" ")
                # else: continue
                for line in split_line:
                    time_data = parse_time_data(line)
                    flight_data = parse_flight_data(line)
                    gate_data = parse_gate_assign_data(line)
                    if time_data:
                        data_type, agent_id, segment_id, time = time_data
                        # Ensure the lists are large enough
                        while len(start_times) <= agent_id:
                            start_times.append([])
                            arrival_times.append([])
                        while len(start_times[agent_id]) <= segment_id:
                            start_times[agent_id].append(None)
                            arrival_times[agent_id].append(None)
                        # Store the time data
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

    # Group flights by agent_ID
    agent_flights = {}
    for agent, (origin, destination), passengers, segment in flights:
        agent_flights.setdefault(agent, []).append((segment, origin, destination, passengers))
    # Group gate by agent_ID and segment
    agent_gates = {}
    for agent_id, segment, gate_id, destination in gates:
        agent_gates.setdefault(agent_id, {}).setdefault(segment, []).append((gate_id, destination))
    # Ensure the visualize directory exists
    output_dir = os.path.dirname(__file__)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a single undirected graph for all agents
    G = nx.Graph()
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # colors for different agents

    plt.figure(figsize=(15, 8))

    # Create sequential nodes for each agent
    pos = {}
    y_spacing = 1.0  # Vertical spacing between agent paths
    max_segments = max(seg[0] for agent_segs in agent_flights.values() for seg in agent_segs)

    # increase distance between each node = 1
    add = [i*4 for i in range(0,len(arrival_times[0]))]
    arrival_times_true = arrival_times
    arrival_times = [[x+y if x != None else None for x,y in zip(a,add)] for a in arrival_times]
    # Adjust start_times similar to arrival_times
    start_times_true = start_times
    start_times = [[x+y if x != None else None for x, y in zip(a, add)] for a in start_times]
    
    for agent, segments in agent_flights.items():
        y_pos = -agent * y_spacing  # Each agent gets its own row
        previous_h = 0
        for seg_num, origin, destination, passengers in sorted(segments, key=lambda x: x[0]):
            # Create unique node names for visualization
            src_node = f"{origin}_{agent}_{seg_num}"
            depart_node = f"{origin}_{agent}_{seg_num}_depart"
            dst_node = f"{destination}_{agent}_{seg_num}"
            
            # distance between origin and destination
            dist_city = dist[city.index(origin), city.index(destination)]
            # Add nodes with their positions
            # Safely get index of city, handle potential errors
            try:
                origin_idx = city.index(origin)
                dest_idx = city.index(destination)
            except ValueError:
                print(f"Warning: City '{origin}' or '{destination}' not found in city list. Using default distance.")
                dist_city = 1.0  # Default distance if city not found
            if seg_num == 1:      
                pos[src_node] = (0, y_pos)
                pos[depart_node] = (0, y_pos)
                pos[dst_node] = (arrival_times[agent][seg_num]/5, y_pos)
            else:
                pos[src_node] = (arrival_times[agent][seg_num-1]/5, y_pos)
                pos[depart_node] = (start_times[agent][seg_num]/5, y_pos)
                pos[dst_node] = (arrival_times[agent][seg_num]/5, y_pos)

            # Add to graph
            G.add_node(src_node)
            G.add_node(depart_node)
            G.add_node(dst_node)
            G.add_edge(src_node, depart_node)
            nx.draw_networkx_edges(G, pos, edgelist=[(src_node, depart_node)], style='dotted', edge_color='black', width=0.5)
            G.add_edge(depart_node, dst_node)
            if passengers == 0:
                # edge_colors.append('gray')
                edge_color = 'gray'
            elif passengers == 4:
                # edge_colors.append('red')
                edge_color = 'red'
            else:   
                # edge_colors.append('black')
                edge_color = 'black'
            nx.draw_networkx_edges(G, pos, edgelist=[(depart_node, dst_node)], style='solid', edge_color=edge_color, width=0.5)    

            # Store original location names for labels
            if seg_num == 1:
                G.nodes[src_node]['label'] = origin
            else:
                G.nodes[src_node]['label'] = ""
            G.nodes[depart_node]['label'] = ""
            G.nodes[dst_node]['label'] = destination

            # Add start time to the label
            if start_times[agent][seg_num] is not None:
                G.nodes[depart_node]['label'] = f"\n{start_times_true[agent][seg_num]}\u2192\n"
            # Get gate_id from agent_ID and segment
            gate_info = agent_gates.get(agent, {}).get(seg_num, [])
            if gate_info:
                gate_id, gate_destination = gate_info[0]
                G.nodes[dst_node]['label'] += f":{gate_id}"
            else:
                G.nodes[dst_node]['label'] += "\nGate: N/A"
            # Add arrival time to the label
            if arrival_times[agent][seg_num] is not None:
                G.nodes[dst_node]['label'] = f"\n{arrival_times_true[agent][seg_num]}\n" + G.nodes[dst_node]['label']
                            
    # Draw edges for each agent with different colors
    for agent, segments in agent_flights.items():
        agent_color = colors[agent % len(colors)]
        
        # Create edges for this agent
        edge_list = []
        edge_labels = {}
        edge_colors = []
        for seg_num, origin, destination, passengers in sorted(segments, key=lambda x: x[0]):
            src_node = f"{origin}_{agent}_{seg_num}"
            depart_node = f"{origin}_{agent}_{seg_num}"
            dst_node = f"{destination}_{agent}_{seg_num}"
            
            edge_list.append((src_node, depart_node))
            edge_list.append((depart_node, dst_node))
            edge_labels[(depart_node, dst_node)] = f"A{agent}-S{seg_num}\n({passengers})"

            if passengers == 0:
                # edge_colors.append('gray')
                edge_color = 'gray'
            elif passengers == 4:
                # edge_colors.append('red')
                edge_color = 'red'
            else:   
                # edge_colors.append('black')
                edge_color = 'black'

        # Draw edges for this agent
        # nx.draw_networkx_edges(G, pos, edgelist=edge_list,
        #                     edge_color=edge_colors, width=0.5)  # Reduce edge thickness


        # Draw edge labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                    font_size=5, font_color='black')

    # Draw nodes and custom labels
    
    nx.draw_networkx_nodes(G, pos, node_size=1, node_color='lightblue')
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=13, font_weight='bold', verticalalignment='bottom')

    plt.title("Sequential Flight Segments for All Agents")
    plt.axis('off')
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, "all_agents_trajectory.svg")
    plt.savefig(output_path, dpi=1000, format="svg", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    visualize()