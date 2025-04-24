# %%
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import sys
import os
import math
# import matplotlib.pyplot as plt
# import pyomo.environ as pyenv
import json
import pprint
# Set the working directory

# os.chdir('/Users/duong/Dropbox/NMSU/NASA_ULI_2022/git-public/evtol-scheduling-rqtime/program_1')

velocity = 100 # mph
e_dischg = 4 # kWh/mile
e_max = 100 # kWh
g_i=250 # kW
city_names = ["lgr", "jfk", "rep", "lima", "dtm", "ttb", "nli"]
nb_edges = len(city_names)*(len(city_names)-1)
dist = np.array([
    [ 0., 10.68, 24.32, 40.51,  8.876, 11.08, 16.6],
    [10.68,  0.,  20.15, 37.19, 12.82,  20.73, 20.81],
    [24.32,  20.15,  0., 17.05, 31.33,  34.96,  39.74],
    [40.51, 37.19, 17.05,  0., 48.12,  50.47,  56.5],
    [ 8.876, 12.82, 31.33, 48.12,  0., 10.63, 8.407],
    [11.08,  20.73,  34.96,  50.47, 10.63,  0.,  12.26],
    [16.6, 20.81,  39.74,  56.5, 8.407,  12.26,  0.]
                                ])


def CalProfit(*answer_set):

    # Load the Excel file and read only the first 8 rows with the first row as the index
    MER_LMP = {
        'MER': pd.read_excel(fr'MER_LMP_Information.xlsx', sheet_name='MER', nrows=7, index_col=0),
        'LMP': pd.read_excel(fr'MER_LMP_Information.xlsx', sheet_name='LMP', nrows=7, index_col=0)
    }

    # Replace index row with ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
    # city_names = ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
    # # distance between city ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
    # dist = np.array(
    #     [
    #         [0.,          10.48113508,  18.51467981,  18.54127528,  5.7385456,    13.03650962,  20.09902115], 
    #         [10.48113508,  0.,           9.15619145,   15.04929904,  10.56004164,  6.52505341,   13.22524436],
    #         [18.51467981,  9.15619145,   0.,           11.47944457,  16.1038714,   6.31668958,   6.33708216 ],
    #         [18.54127528,  15.04929904,  11.47944457,  0.,           13.32823709,  8.5872692,    6.16996146 ],
    #         [5.7385456,    10.56004164,  16.1038714,   13.32823709,  0.,           9.86908414,   16.01052999],
    #         [13.03650962,  6.52505341,   6.31668958,   8.5872692,    9.86908414,   0.,           7.31033037 ],
    #         [20.09902115,  13.22524436,  6.33708216,   6.16996146,   16.01052999,  7.31033037,   0.         ]
    #         ]
    #     )
    

    MER_LMP['MER'].index = city_names
    MER_LMP['LMP'].index = city_names

    # Naming index name as 'city'
    MER_LMP['MER'].index.name = 'city'
    MER_LMP['LMP'].index.name = 'city'

    def fly_time(i, j):
        return dist[i][j]/velocity * 60




    
    # dist = np.round(dist)
    min_trav_time = np.min(dist[np.nonzero(dist)])/velocity*60
    # min_trav_time=2.2
    N_steps = int(np.floor(180/min_trav_time))
    # Create a DataFrame with the distance matrix
    dist = pd.DataFrame(dist, index=city_names, columns=city_names)
    # Read output_2352.txt
    # with open(fr'solver_to_answer_set.txt', 'r') as file:
    #     content = file.read()

    # Remove content until the first newline character, including the newline itself

    # Separate file content into a list of strings where separation is a blank space
    list_of_strings = answer_set[0]

    # Get only strings that have the substring 'dl(start_v' or 'dl(arrival_v' or 'as_w'
    # filtered_strings = [s for s in list_of_strings if 'as' in s or 'dl(arrival_v' in s]
    filtered_strings = list_of_strings
    # Regular expressions to extract components from the predicates
    arrival_v_pattern = re.compile(r'dl\(arrival\((\d+),\((\w+),(\w+)\),(\d+)\),(\d+)\)')
    as_w_pattern = re.compile(r'as_w\((\d+),\((\w+),(\w+)\),(\d+),(\d+)\)')
    as_pattern = re.compile(r'as\((\d+),\((\w+),(\w+)\),(\d+),(\d+)\)')
    passengers_served_pattern = re.compile(r'passengers_served\((\d+),\((\w+),(\w+)\)\)')
    start_v_pattern = re.compile(r'dl\(start\((\d+),\((\w+),(\w+)\),(\d+)\),(\d+)\)')
    # start_v_2_pattern = re.compile(r'dl\(start\((\d+),\((\w+),(\w+)\),(\d+)\),(\d+)\)')
    start_v_2_pattern = re.compile(r'dl\(start\((\d+),\((\w+),(\w+)\),(\d+),(\d+)\),(\d+)\)')
    pot_revenue_pattern = re.compile(r'pot_revenue\(\((\w+),(\w+)\),(\d+),(\d+)\)')

    # Nested dictionary to store the data
    nested_dict = defaultdict(lambda: defaultdict(dict))
    sort_asp = defaultdict(lambda: defaultdict(dict))
    # Initialize the dictionary to store the summation of weights
    weight_summation = defaultdict(lambda: {"total_served": 0, "from_asp_solution": []})
    dict_check_asp = defaultdict(lambda: defaultdict(dict))

    pot_revenue = defaultdict(lambda: defaultdict(dict))

    # Process each filtered string
    for s in filtered_strings:
        if 'dl(arrival' in s:
            match = arrival_v_pattern.match(s)
            if match:
                agent_id, vertiport_departure,vertiport_arrival, step_id, time_arrival = match.groups()
                step_id = int(step_id)
                # nested_dict[agent_id][step_id]['arrival'] = s

                # nested_dict[agent_id][step_id]['TIME_ARRIVAL'] = int(time_arrival)
                # nested_dict[agent_id][step_id-1]['next_arrival_v'] = s
                # nested_dict[agent_id][step_id-1]['next_TIME_ARRIVAL'] = int(time_arrival)
                flight_time = dist[vertiport_departure][vertiport_arrival]*0.6
                charge_time = flight_time*24/25
                dict_check_asp[int(agent_id)][step_id]['flight_time'] = flight_time
                dict_check_asp[int(agent_id)][step_id][2] = s

                dict_check_asp[int(agent_id)][step_id+1]["start_charge_asp"] = int(time_arrival)
                dict_check_asp[int(agent_id)][step_id]["destination"] = vertiport_arrival
                if "start_charge" in dict_check_asp[int(agent_id)][step_id]:                 
                    dict_check_asp[int(agent_id)][step_id]["stop_charge_asp"] = dict_check_asp[int(agent_id)][step_id]["start_charge"] + np.round(dist[vertiport_departure][vertiport_arrival]*24/25)
                charge_time_residual = np.round(dist[vertiport_departure][vertiport_arrival]*24/25) - (dist[vertiport_departure][vertiport_arrival]*24/25) # plus into horizon. 
        elif 'passengers_served' in s:
            match = passengers_served_pattern.match(s)
            if match:
                w, vertiport_departure, vertiport_arrival = match.groups()
                weight_summation[(vertiport_departure, vertiport_arrival)]["from_asp_solution"].append(w)
        elif 'as(' in s:
            match = as_pattern.match(s)
            if match:
                agent_id, vertiport_departure, vertiport_arrival, weight, step_id = match.groups()
                dict_check_asp[int(agent_id)][int(step_id)]['revenue'] = int(weight) * dist.loc[vertiport_departure, vertiport_arrival] * 0.5
                step_id = int(step_id)
                dict_check_asp[int(agent_id)][int(step_id)]['weight'] = int(weight)
                # nested_dict[agent_id][step_id-1]['as'] = s
                nested_dict[agent_id][step_id-1]['VERTIPORT'] = vertiport_departure
                nested_dict[agent_id][step_id-1]['DISTANCE'] = dist[vertiport_departure][ vertiport_arrival]
                sort_asp[int(agent_id)][int(step_id)]['route'] = s


                
                dict_check_asp[int(agent_id)][step_id][0] = s

        elif 'start' in s:
            match = start_v_pattern.match(s)
            if match:
                agent_id, vertiport_start, vertiport_arrival, step_id, time_start = match.groups()
                dict_check_asp[int(agent_id)][int(step_id)][1] = s
                dict_check_asp[int(agent_id)][int(step_id)]['departure_time_asp'] = int(time_start)
                dict_check_asp[int(agent_id)][int(step_id)]['charge_at_vertiport'] = vertiport_start
        elif 'pot_revenue' in s:
            match = pot_revenue_pattern.match(s)
            if match:
                vertiport_start, vertiport_arrival, revenue, step_id = match.groups()
                pot_revenue[int(step_id)][(vertiport_start, vertiport_arrival)]= revenue


    # Function to get MER and LMP values for a given vertiport and time range
    def get_mer_lmp(vertiport, time_arrival, time_depart):
        col = math.floor(time_arrival/10)
        next_col = math.ceil(time_depart/10)
        mer_values = MER_LMP['MER'].loc[vertiport][col:(next_col)].tolist()
        lmp_values = MER_LMP['LMP'].loc[vertiport][col:(next_col)].tolist()
        col_out = MER_LMP['MER'].columns[col:next_col].tolist()
        return mer_values, lmp_values, col_out


    stationed_route_count = 0  # Initialize counter
    empty_flight_count = 0
    # Loop through the nested dictionary to recalculate the timing
    for agent_id in range(0, len(dict_check_asp)):
        path = dict_check_asp[agent_id]
        for step_id in range(0, len(path)):
            data = path[step_id]
            if step_id == 1:
                data['current_battery'] = 25
                arrival_time = data['flight_time']
                dict_check_asp[agent_id][step_id+1]['previous_arrival_time'] = arrival_time
            if 'flight_time' in data and data['current_battery'] >= data['flight_time']:
                data['remain_battery'] = data['current_battery'] - data['flight_time']
                data['charge_time'] = 0
            elif 'flight_time' in data:
                data['aftercharge_battery'] = data['flight_time']
                data['charge_time'] = (data['aftercharge_battery'] - data['current_battery'])*24/25
                data['remain_battery'] = data['aftercharge_battery'] - data['flight_time']
            if 'remain_battery' in data:
                dict_check_asp[agent_id][step_id+1]['current_battery'] = data['remain_battery']
            
            if 0 in data and 'previous_arrival_time' in data:
                data['departure_time'] = data['previous_arrival_time'] + data['charge_time']
                data['start_charge'] = data['previous_arrival_time']
                data['stop_charge'] = data['start_charge'] + data['charge_time']

            #     else:
                    # path[step_id]['departure_time'] = data['departure_time_asp']
            if step_id != 0 and 'departure_time' in data:
                arrival_time = data['departure_time'] + data['flight_time']
                dict_check_asp[agent_id][step_id+1]['previous_arrival_time'] = arrival_time

            if 'destination' in data and 'charge_at_vertiport' in data and data['destination'] == data['charge_at_vertiport']:
                stationed_route_count += 1  # Increment counter if both keys are present
            if 'destination' in data and 'charge_at_vertiport' in data and data['destination'] != data['charge_at_vertiport'] and data['weight'] == 0:
                empty_flight_count += 1
    # Loop through the nested dictionary to add MER and LMP lists
    for agent_id, steps in dict_check_asp.items():
        for step_id, data in steps.items():
            if 'charge_time' in data and data['charge_time'] > 0:
                vertiport = data['charge_at_vertiport']
                destination = data['destination']
                start_charge = data['start_charge']
                stop_charge = data['stop_charge']
                mer_values, lmp_values, col = get_mer_lmp(vertiport, start_charge, stop_charge)
                data['MER'] = mer_values
                data['LMP'] = lmp_values
                data['MER+LMP'] = [m*185 + l for m,l in zip(mer_values, lmp_values)]
                # if len(data['MER+LMP']) > 1:
                #     data['MER+LMP_variance'] = np.var(data['MER+LMP'])
                # else:
                #     data['MER+LMP_variance'] = 0
                data['col'] = col
                if len(col) > 1:
                    for i in range(0, len(col)):
                        if i == 0:
                            data['CHARGING_COST'] = ((col[0] - data['start_charge'])/60) * ((data['LMP'][0])/1000) * 250
                            data['EMISSION_COST'] = ((col[0] - data['start_charge'])/60) * ((data['MER'][0]*185)/1000) * 250                   
                        elif i == len(col)-1:
                            data['CHARGING_COST'] += ((data['stop_charge']-col[i-1])/60) * ((data['LMP'][i])/1000) * 250
                            data['EMISSION_COST'] += ((data['stop_charge']-col[i-1])/60) * ((data['MER'][i]*185)/1000) * 250
                        else:
                            data['CHARGING_COST'] += (10/60) * ((data['LMP'][i])/1000) * 250
                            data['EMISSION_COST'] += (10/60) * ((data['MER'][i]*185)/1000) * 250
                else:
                    data['CHARGING_COST'] = (data['charge_time']/60) * ((data['LMP'][0])/1000) * 250
                    data['EMISSION_COST'] = (data['charge_time']/60) * ((data['MER'][0]*185)/1000) * 250
                # data['CHARGING_COST'] = ((dist[vertiport][destination])*24/(25*100)) * (np.mean(data['LMP'])/1000) * 250
                # data['EMISSION_COST'] = ((dist[vertiport][destination])*24/(25*100)) * (np.mean(data['MER'])/1000) * 185 * 250

    # Print the nested dictionary for verification
    
    # pprint.pprint(nested_dict)

    # Sum revenue and cost
    total_revenue = 0
    total_em_cost = 0
    total_chg_cost = 0
    for agent_id, steps in dict_check_asp.items():
        for step_id, data in steps.items():
            if 'CHARGING_COST' in data:
                if not np.isnan(data['CHARGING_COST']):
                    total_chg_cost += data['CHARGING_COST']
                # else:
                    # print(agent_id, step_id)
            if 'EMISSION_COST' in data:
                if not np.isnan(data['EMISSION_COST']):
                    total_em_cost += data['EMISSION_COST']
                # else:
                #     print(agent_id, step_id)
                #     print( data   )             
            if 'revenue' in data:
                total_revenue += data['revenue']
                

    # compute obj value
    # -- test 
    # pprint.pprint(nested_dict_1)
    # with open(fr'answer_set_sorted.txt', 'w') as txt_file:
    #     original_stdout = sys.stdout  # Save a reference to the original standard output
    #     sys.stdout = txt_file  # Redirect standard output to the file
    #     try:
    #         try:
    #             pprint.pprint(dict_check_asp)
    #             print(f'revenue={total_revenue}')
    #             print(f'em_cost={total_em_cost}')
    #             print(f'chg_cost={total_chg_cost}')
    #             print(f'profit={total_revenue - total_em_cost - total_chg_cost}')
    #         except Exception as e:
    #             print(f"An error occurred: {e}")
    #     finally:
    #         sys.stdout = original_stdout  # Reset standard output to its original value

    # with open(fr'answer_set_sorted.txt', 'w') as txt_file:
    #     original_stdout = sys.stdout  # Save a reference to the original standard output
    #     sys.stdout = txt_file  # Redirect standard output to the file
    #     try:
    #         try:
    #             print(f'revenue={total_revenue}')
    #             print(f'profit={total_revenue - total_em_cost - total_chg_cost}')     
    #             print(f'em_cost={total_em_cost}')
    #             print(f'chg_cost={total_chg_cost}')

    #             pprint.pprint(sort_asp)

    #         except Exception as e:
    #             print(f"An error occurred: {e}")
    #     finally:
    #         sys.stdout = original_stdout  # Reset standard output to its original value

    with open(fr'results/result_answer_set_sorted_detail.txt', 'w') as txt_file:
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = txt_file  # Redirect standard output to the file
        try:
            try:
                print(f'revenue={total_revenue}')
                print(f'profit={total_revenue - total_em_cost - total_chg_cost}')
                print(f'em_cost={total_em_cost}')
                print(f'chg_cost={total_chg_cost}')
                pprint.pprint(dict_check_asp)

            except Exception as e:
                print(f"An error occurred: {e}")
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value
            

    answer_set_print = ""
    with open(fr'results/result_answer_set_sorted.lp', 'w') as txt_file:
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = txt_file  # Redirect standard output to the file
        try:
            try:
                for agent, steps in dict(sorted(sort_asp.items())).items():
                    for step, route in steps.items():
                        print(route['route']+'.')
                        answer_set_print += route['route']+'.\n'

            except Exception as e:
                print(f"An error occurred: {e}")
        finally:
            sys.stdout = original_stdout  # Reset standard output to its original value
            
    print(f'revenue={total_revenue}')
    print(f'profit={total_revenue - total_em_cost - total_chg_cost}')
    print(f'em_cost={total_em_cost}')
    print(f'chg_cost={total_chg_cost}')
    print(f'Number of segments that agents stationed: {stationed_route_count}')
    print(f'Number of flights that agents carry zero customer: {empty_flight_count}')
    return total_revenue, total_em_cost, total_chg_cost, total_revenue-total_em_cost -total_chg_cost

def EstimateMinAgents(horizon, b_init, demand_cust, aircraft_capacity):
    """
    b_init in charging minute
    """
    minimum_flights_per_edge = math.ceil(demand_cust/aircraft_capacity)
    require_operation_time = np.sum(dist)*(60/velocity + e_dischg*60/g_i)
    require_operation_time_serve_all_cust = minimum_flights_per_edge * require_operation_time
    min_number_agents = math.ceil(require_operation_time_serve_all_cust/(horizon+b_init))
    min_number_segments = math.ceil((minimum_flights_per_edge*nb_edges)/min_number_agents)
    return min_number_agents, min_number_segments

if __name__ == "__main__":
    print(EstimateMinAgents(360, 60, 30, 4))
# %%
