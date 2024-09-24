# %%
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import sys
import os
import math
import matplotlib.pyplot as plt
import pyomo.environ as pyenv
import json



velocity = 100
# Load the Excel file and read only the first 8 rows with the first row as the index
MER_LMP = {
    'MER': pd.read_excel('MER_LMP_Information.xlsx', sheet_name='MER', nrows=7, index_col=0),
    'LMP': pd.read_excel('MER_LMP_Information.xlsx', sheet_name='LMP', nrows=7, index_col=0)
}

# Replace index row with ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
city_names = ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
MER_LMP['MER'].index = city_names
MER_LMP['LMP'].index = city_names

# Naming index name as 'city'
MER_LMP['MER'].index.name = 'city'
MER_LMP['LMP'].index.name = 'city'

def fly_time(i, j):
    return dist[i][j]/velocity * 60

# distance between city ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
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

min_trav_time = np.min(dist[np.nonzero(dist)])/velocity*60
# min_trav_time=2.2
N_steps = int(np.floor(180/min_trav_time))
# Create a DataFrame with the distance matrix
dist = pd.DataFrame(dist, index=city_names, columns=city_names)
# Read output_2352.txt
with open('input_answer_set.txt', 'r') as file:
    content = file.read()
# Separate file content into a list of strings where separation is a blank space
list_of_strings = content.split()

# Get only strings that have the substring 'dl(start_v' or 'dl(arrival_v' or 'as_w'
# filtered_strings = [s for s in list_of_strings if 'as' in s or 'dl(arrival_v' in s]
filtered_strings = list_of_strings
# Regular expressions to extract components from the predicates
arrival_v_pattern = re.compile(r'dl\(arrival\((\d+),(\w+),(\d+)\),(\d+)\)')
as_w_pattern = re.compile(r'as_w\((\d+),\((\w+),(\w+)\),(\d+),(\d+)\)')
as_pattern = re.compile(r'as\((\d+),\((\w+),(\w+)\),(\d+)\)')
passengers_served_pattern = re.compile(r'passengers_served\((\d+),\((\w+),(\w+)\)\)')
start_v_pattern = re.compile(r'dl\(start_v\((\d+),(\w+),(\d+)\),(\d+)\)')
# start_v_2_pattern = re.compile(r'dl\(start\((\d+),\((\w+),(\w+)\),(\d+)\),(\d+)\)')
start_v_2_pattern = re.compile(r'dl\(start\((\d+),\((\w+),(\w+)\),(\d+),(\d+)\),(\d+)\)')
'''
(Keep this comment!)
dl(arrival_v({AGENT_ID},{VERTIPORT_ARRIVAL},{STEP_ID}),{TIME_ARRIVAL})
as_w({AGENT_ID},({VERTIPORT_DEPARTURE},{VERTIPORT_ARRIVAL}),{STEP_ID},{WEIGHT})
as({AGENT_ID},({VERTIPORT_DEPARTURE},{VERTIPORT_ARRIVAL}),{STEP_ID})
'''
''' 
create nested directory where each AGENT_ID has a directory; each STEP_ID>0 has a directory of 4 predicates:
dl(arrival_v({AGENT_ID},{VERTIPORT_DEPARTURE},{STEP_ID}-1),{TIME_DEPARTURE})
dl(arrival_v({AGENT_ID},{VERTIPORT_DEPARTURE},{STEP_ID}),{TIME_DEPARTURE})
as_w({AGENT_ID},({VERTIPORT_DEPARTURE},{VERTIPORT_ARRIVAL}),{STEP_ID},{WEIGHT})
as({AGENT_ID},({VERTIPORT_DEPARTURE},{VERTIPORT_ARRIVAL}),{STEP_ID})

if STEP_ID = 0 then we have 3 predicates:
dl(arrival_v({AGENT_ID},{VERTIPORT_DEPARTURE},{STEP_ID}),{TIME_DEPARTURE})
as_w({AGENT_ID},({VERTIPORT_DEPARTURE},{VERTIPORT_ARRIVAL}),{STEP_ID},{WEIGHT})
as({AGENT_ID},({VERTIPORT_DEPARTURE},{VERTIPORT_ARRIVAL}),{STEP_ID})
passengers_served({WEIGHT}, ({VERTIPORT_DEPARTURE}, {VERTIPORT_ARRIVAL}))
'''
# Nested dictionary to store the data
nested_dict = defaultdict(lambda: defaultdict(dict))
nested_dict_1 = defaultdict(lambda: defaultdict(dict))
# Initialize the dictionary to store the summation of weights
weight_summation = defaultdict(lambda: {"total_served": 0, "from_asp_solution": []})
dict_check_asp = defaultdict(lambda: defaultdict(dict))
# Process each filtered string
for s in filtered_strings:
    if 'dl(arrival' in s:
        match = arrival_v_pattern.match(s)
        if match:
            agent_id, vertiport_arrival, step_id, time_arrival = match.groups()
            step_id = int(step_id)
            nested_dict[agent_id][step_id]['arrival'] = s

            nested_dict[agent_id][step_id]['TIME_ARRIVAL'] = int(time_arrival)
            nested_dict[agent_id][step_id-1]['next_arrival_v'] = s
            nested_dict[agent_id][step_id-1]['next_TIME_ARRIVAL'] = int(time_arrival)
            dict_check_asp[int(agent_id)][step_id][2] = s                
    elif 'as_w' in s:
        match = as_w_pattern.match(s)
        if match:
            agent_id, vertiport_departure, vertiport_arrival, step_id, weight = match.groups()
            if int(weight) == 8: 
                weight = 4
            elif int(weight) == 10: 
                weight = 2
            step_id = int(step_id)
            nested_dict[agent_id][step_id]['as_w'] = s
            nested_dict[agent_id][step_id]['REVENUE'] = int(weight) * dist.loc[vertiport_departure, vertiport_arrival] * 0.5
            nested_dict_1[int(agent_id)][step_id]['passengers'] = s
            
            # Update the summation dictionary
            weight_summation[(vertiport_departure, vertiport_arrival)]['total_served'] += int(weight)
            
            dict_check_asp[int(agent_id)][step_id][0] = s
    elif 'passengers_served' in s:
        match = passengers_served_pattern.match(s)
        if match:
            w, vertiport_departure, vertiport_arrival = match.groups()
            weight_summation[(vertiport_departure, vertiport_arrival)]["from_asp_solution"].append(w)
    elif 'as(' in s:
        match = as_pattern.match(s)
        if match:
            agent_id, vertiport_departure, vertiport_arrival, step_id = match.groups()
            step_id = int(step_id)
            # nested_dict[agent_id][step_id-1]['as'] = s
            nested_dict[agent_id][step_id-1]['VERTIPORT'] = vertiport_departure
            nested_dict[agent_id][step_id-1]['DISTANCE'] = dist[vertiport_departure][ vertiport_arrival]
            
            nested_dict_1[int(agent_id)][step_id]['route'] = s
            nested_dict_1[int(agent_id)][step_id]['charge_fly_time'] = fly_time(vertiport_departure, vertiport_arrival)
            # if step_id == 0:
            #     nested_dict[agent_id][step_id-1]['TIME_ARRIVAL'] = 0
            #     nested_dict_1[int(agent_id)][step_id]['start_milp'] = math.ceil((fly_time(vertiport_departure, vertiport_arrival))/min_trav_time) * min_trav_time
            #     nested_dict_1[int(agent_id)][step_id]['start'] = fly_time(vertiport_departure, vertiport_arrival)
            # else:
                
            #     nested_dict_1[int(agent_id)][step_id]['start'] = nested_dict_1[int(agent_id)][step_id-1]['arrival'] + fly_time(vertiport_departure, vertiport_arrival)
            #     nested_dict_1[int(agent_id)][step_id]['start_milp'] = math.ceil((nested_dict_1[int(agent_id)][step_id-1]['start_charging']+ fly_time(vertiport_departure, vertiport_arrival))/min_trav_time) * (min_trav_time)

            # nested_dict_1[int(agent_id)][step_id]['arrival'] = nested_dict_1[int(agent_id)][step_id]['start'] + nested_dict_1[int(agent_id)][step_id]['charge_fly_time']
            # nested_dict_1[int(agent_id)][step_id]['arrival_milp'] = nested_dict_1[int(agent_id)][step_id]['start_milp'] + nested_dict_1[int(agent_id)][step_id]['charge_fly_time']
            # nested_dict_1[int(agent_id)][step_id]['start_charging'] = math.ceil((nested_dict_1[int(agent_id)][step_id]['arrival_milp'] /min_trav_time)) * min_trav_time
            
            dict_check_asp[int(agent_id)][step_id][0] = s
    # elif 'start' in s:
    #     match = start_v_pattern.match(s)
    #     if match:
    #         agent_id, vertiport_start, step_id, time_start = match.groups()
    #         dict_check_asp[int(agent_id)][int(step_id)][1] = s
    elif 'start' in s:
        match = start_v_2_pattern.match(s)
        if match:
            agent_id, vertiport_start, vertiport_arrival, step_id, weight, time_start = match.groups()
            dict_check_asp[int(agent_id)][int(step_id)][1] = s
# Loop through the nested dictionary
# for agent_id, steps in nested_dict.items():
#     for step_id, data in steps.items():
#         if 'next_TIME_ARRIVAL' in data and 'DISTANCE' in data:
#             data['FLY_TIME'] = data['DISTANCE'] * 0.6 
#             data['TIME_STOP_CHARGING'] = data['TIME_ARRIVAL'] + data['FLY_TIME']
#             data['TIME_DEPART'] = data['next_TIME_ARRIVAL'] - data['DISTANCE'] * 0.6

            
# Function to get MER and LMP values for a given vertiport and time range
def get_mer_lmp(vertiport, time_arrival, time_depart):
    col = math.floor(time_arrival/10)
    next_col = math.ceil(time_depart/10)
    mer_values = MER_LMP['MER'].loc[vertiport][col:next_col].tolist()
    lmp_values = MER_LMP['LMP'].loc[vertiport][col:next_col].tolist()
    col_out = MER_LMP['MER'].columns[col:next_col].tolist()
    return mer_values, lmp_values, col_out

# Loop through the nested dictionary to add MER and LMP lists
for agent_id, steps in nested_dict.items():
    for step_id, data in steps.items():
        if 'VERTIPORT' in data and 'TIME_ARRIVAL' in data and 'TIME_DEPART' in data:
            vertiport = data['VERTIPORT']
            time_arrival = data['TIME_ARRIVAL']
            time_depart = data['TIME_DEPART']
            mer_values, lmp_values, col = get_mer_lmp(vertiport, time_arrival, time_depart)
            data['MER'] = mer_values
            data['LMP'] = lmp_values
            data['MER+LMP'] = [m*185 + l for m,l in zip(mer_values, lmp_values)]
            if len(data['MER+LMP']) > 1:
                data['MER+LMP_variance'] = np.var(data['MER+LMP'])
            else:
                data['MER+LMP_variance'] = 0
            data['col'] = col
            data['CHARGING_COST'] = (data['FLY_TIME']/60) * (np.mean(data['LMP'])/1000) * 250
            data['EMISSION_COST'] = (data['FLY_TIME']/60) * (np.mean(data['MER'])/1000) * 185 * 250
            

# Print the nested dictionary for verification
import pprint
# pprint.pprint(nested_dict)

# Sum revenue and cost
total_revenue = 0
total_em_cost = 0
total_chg_cost = 0
for agent_id, steps in nested_dict.items():
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
        if 'REVENUE' in data:
            total_revenue += data['REVENUE']
            

# compute obj value
# -- test 
# pprint.pprint(nested_dict_1)
with open('dict_check_asp.txt', 'w') as txt_file:
    original_stdout = sys.stdout  # Save a reference to the original standard output
    sys.stdout = txt_file  # Redirect standard output to the file
    try:
        try:
            pprint.pprint(dict_check_asp)
            print(f'revenue={total_revenue}')
            print(f'em_cost={total_em_cost}')
            print(f'chg_cost={total_chg_cost}')
            print(f'profit={total_revenue - total_em_cost - total_chg_cost}')
        except Exception as e:
            print(f"An error occurred: {e}")
    finally:
        sys.stdout = original_stdout  # Reset standard output to its original value
