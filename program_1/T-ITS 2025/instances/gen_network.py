
import csv

import random
# import networkx as nx
import copy
import numpy as np
import math
import sys
import pandas as pd
class GenerateMapGivenPath:
    def __init__(self, nAgents=10, nRquests=20, timestep=30): #max request = 20
        '''init predicate'''
        self.fName = 'network_NY_update.lp'
        self.nRquests = nRquests
        self.nAgents = nAgents

        self.agent_id = [i for i in range(0, self.nAgents)]




        # # NY
        # self.vPort_id = ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
        # self.vPort_print = "jfk;lga;teb;ryend;cri;cimbl;dandy"
        # self.dist = pd.DataFrame(
        # [
        # [0.,          10.48113508,  18.51467981,  18.54127528,  5.7385456,    13.03650962,  20.09902115], 
        # [10.48113508,  0.,           9.15619145,   15.04929904,  10.56004164,  6.52505341,   13.22524436],
        # [18.51467981,  9.15619145,   0.,           11.47944457,  16.1038714,   6.31668958,   6.33708216 ],
        # [18.54127528,  15.04929904,  11.47944457,  0.,           13.32823709,  8.5872692,    6.16996146 ],
        # [5.7385456,    10.56004164,  16.1038714,   13.32823709,  0.,           9.86908414,   16.01052999],
        # [13.03650962,  6.52505341,   6.31668958,   8.5872692,    9.86908414,   0.,           7.31033037 ],
        # [20.09902115,  13.22524436,  6.33708216,   6.16996146,   16.01052999,  7.31033037,   0.  ]
        # ],
        # self.vPort_id,
        # self.vPort_id
        # )
        # NY update
        self.vPort_id = ["lgr", "jfk", "rep", "lima", "dtm", "ttb", "nli"]
        self.vPort_print = "lgr;jfk;rep;lima;dtm;ttb;nli"
        self.dist = pd.DataFrame(
                                    [[ 0., 10.68, 24.32, 40.51,  8.876, 11.08, 16.6],
                                    [10.68,  0.,  20.15, 37.19, 12.82,  20.73, 20.81],
                                    [24.32,  20.15,  0., 17.05, 31.33,  34.96,  39.74],
                                    [40.51, 37.19, 17.05,  0., 48.12,  50.47,  56.5],
                                    [ 8.876, 12.82, 31.33, 48.12,  0., 10.63, 8.407],
                                    [11.08,  20.73,  34.96,  50.47, 10.63,  0.,  12.26],
                                    [16.6, 20.81,  39.74,  56.5, 8.407,  12.26,  0.]],
            self.vPort_id,
            self.vPort_id
        )
        # self.distance = pd.DataFrame(
        #                             [[ 0, 10, 19, 19,  6, 13, 20],
        #                             [10,  0,  9, 15, 11,  7, 13],
        #                             [19,  9,  0, 11, 16,  6,  6],
        #                             [19, 15, 11,  0, 13,  9,  6],
        #                             [ 6, 11, 16, 13,  0, 10, 16],
        #                             [13,  7,  6,  9, 10,  0,  7],
        #                             [20, 13,  6,  6, 16,  7,  0]],
        #     self.vPort_id,
        #     self.vPort_id
        # )
        velocity = 100
        E_dischg = 4
        g_i = 250
        # velocity : mph
        # g_i : charging rate kw
        # E_dischg = 4 : rate of discharge kwh/mile
        # E_max = 100 : energy capacity of of eVTOL kwh; UAM evtol: 40-100 miles range
        # g_i = 250
        # LMP -> $/mwh
        # MER -> tons/mwh
        #
        # TN
        # self.vPort_id = ["lbm", "mgt", "tci", "mbf", "crj"]
        # self.vPort_print = 'lbm;mgt;tci;mbf;crj'
        # self.dist = pd.DataFrame(
        #     [[0, 132, 218, 88, 108], 
        #      [132, 0, 100, 103, 61],
        #      [218, 100, 0, 203, 160],
        #      [88, 103, 203, 0, 45],
        #      [108, 61, 160, 45, 0]],
        #     self.vPort_id,
        #     self.vPort_id
        #     )
        # velocity = 286
        # E_dischg = 4
        # g_i = 250
        ## --- integer
        self.flight_time = np.round(self.dist*(60/velocity)).astype(int)
        self.charge_time = np.round(self.dist*E_dischg*60/g_i).astype(int)
        
        # change discharge rate from 4 to 1
        self.dischg_rate_print = [[i, 4] for i in range(0, self.nAgents)]


        
        self.ch_rate_print = [[i, 1] for i in self.vPort_id]
        self.distance_print = [[(i, j), int(np.round(self.dist[i][j]))] for i in self.vPort_id for j in self.vPort_id ]
        # self.segment_distance_print = [[(i, j), self.segment_distance[i][j]] for i in self.vPort_id for j in self.vPort_id ]
        self.edge_print = [(i, j) for i in self.vPort_id for j in self.vPort_id if i != j]
        # self.edge_print = [[k] + i for k, i in zip(range(0, len(self.edge_print)), self.edge_print)] 
        
        ## --- integer
        self.flight_time_print = [[(self.vPort_id[i], self.vPort_id[j]), int(self.flight_time[self.vPort_id[i]][self.vPort_id[j]])] for i in range(len(self.vPort_id)) for j in range(len(self.vPort_id)) ]
        
        self.charge_time_print = [[(self.vPort_id[i], self.vPort_id[j]), int(self.charge_time[self.vPort_id[i]][self.vPort_id[j]])] for i in range(len(self.vPort_id)) for j in range(len(self.vPort_id)) ]
        
        ## --- float
        # self.flight_time_print = [[(self.vPort_id[i], self.vPort_id[j]), str(f'"{self.flight_time[i][j]}"')] for i in range(len(self.vPort_id)) for j in range(len(self.vPort_id)) ]
        
        # self.charge_time_print = [[(self.vPort_id[i], self.vPort_id[j]), str(f'"{self.charge_time[i][j]}"')] for i in range(len(self.vPort_id)) for j in range(len(self.vPort_id)) ]        
        
    def Init_loc(self):
        self.init_loc_print = []
        for i in range(0, self.nAgents):
            self.init_loc_print += [[i, random.choice(self.vPort_id)]]
    
    
    def run(self):

        self.Init_loc()

        
        f = open(self.fName,"w+")

        #vertiport
        f.write('vertiport' + str(tuple(self.vPort_print)).replace("'","").replace(', ','') + '.\n')
                
        # #ch_rate
        # f.write('% 1 MW\n')
        # for i in self.ch_rate_print:
        #     f.write('ch_rate' + str(tuple(i)).replace("'","") + '.\n')       

        #distance
        f.write('% distance= miles/10\n')
        for i in self.distance_print:
            f.write('distance' + str(tuple(i)).replace("'","") + '.\n')
        
        for i in self.flight_time_print:
            f.write('flight_time' + str(tuple(i)).replace("'","") + '.\n')

        for i in self.charge_time_print:
            f.write('charge_time' + str(tuple(i)).replace("'","") + '.\n')
        # f.write('%segment distance = distance/5.\n')    
        # for i in self.segment_distance_print:
            # f.write('segment_distance' + str(tuple(i)).replace("'","") + '.\n')        

        #edge
        # for i in self.edge_print:
        #     f.write('edge' + str(tuple(i)).replace("'","") + '.\n')              
        for i in self.vPort_id:
            for j in self.vPort_id:
                f.write('edge' + '((' + i + ',' + j + ')).\n')   
        f.close()
    
            

# def GenerateRandomMap(nAgent, nVertices, nEdge, fName):
if __name__ == '__main__':


    a = GenerateMapGivenPath()
    a.run()

