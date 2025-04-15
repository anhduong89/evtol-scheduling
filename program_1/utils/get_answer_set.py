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

def get_answer_set(*answer_set):
    velocity = 100
    # Load the Excel file and read only the first 8 rows with the first row as the index
    MER_LMP = {
        'MER': pd.read_excel(fr'MER_LMP_Information.xlsx', sheet_name='MER', nrows=7, index_col=0),
        'LMP': pd.read_excel(fr'MER_LMP_Information.xlsx', sheet_name='LMP', nrows=7, index_col=0)
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

    # dist = np.round(dist)
    min_trav_time = np.min(dist[np.nonzero(dist)])/velocity*60
    # min_trav_time=2.2
    N_steps = int(np.floor(180/min_trav_time))
    # Create a DataFrame with the distance matrix
    dist = pd.DataFrame(dist, index=city_names, columns=city_names)
    # Read output_2352.txt
    # with open(fr'solver_to_answer_set.txt', 'r') as file:
    #     content = file.read()
    content = str(answer_set)
    # Separate file content into a list of strings where separation is a blank space
    list_of_strings = content.split()

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
    out_string = ""
    pot_revenue = defaultdict(lambda: defaultdict(dict))
    with open(fr'results/result_answer_set.lp', 'w') as txt_file:
    # Process each filtered string
        for s in filtered_strings:
            if 'as(' in s:
                match = as_pattern.match(s)
                if match:
                    txt_file.write(f"{match.group()}.\n")
    # with open(fr'results/result.lp', 'w') as txt_file:
    #     for s in filtered_strings:
    #         txt_file.write(f"{s} ")



if __name__ == "__main__":
    get_answer_set()
# %%
