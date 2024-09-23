
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
        self.fName = 'network_NY.lp'
        self.nRquests = nRquests
        self.nAgents = nAgents

        self.agent_id = [i for i in range(0, self.nAgents)]




        # NY
        self.vPort_id = ["jfk", "lga", "teb", "ryend", "cri", "cimbl", "dandy"]
        self.vPort_print = "jfk;lga;teb;ryend;cri;cimbl;dandy"
        self.distance = pd.DataFrame(
                                    [[ 0, 10, 19, 19,  6, 13, 20],
                                    [10,  0,  9, 15, 11,  7, 13],
                                    [19,  9,  0, 11, 16,  6,  6],
                                    [19, 15, 11,  0, 13,  9,  6],
                                    [ 6, 11, 16, 13,  0, 10, 16],
                                    [13,  7,  6,  9, 10,  0,  7],
                                    [20, 13,  6,  6, 16,  7,  0]],
            self.vPort_id,
            self.vPort_id
        )
        
        # TN
        # self.vPort_id = ["lbm", "mgt", "tci", "mbf", "crj"]
        # self.vPort_print = 'lbm;mgt;tci;mbf;crj'
        # self.distance = pd.DataFrame(
        #     [[0, 132, 218, 88, 108], 
        #      [132, 0, 100, 103, 61],
        #      [218, 100, 0, 203, 160],
        #      [88, 103, 203, 0, 45],
        #      [108, 61, 160, 45, 0]],
        #     self.vPort_id,
        #     self.vPort_id
        #     )

        # change discharge rate from 4 to 1
        self.dischg_rate_print = [[i, 4] for i in range(0, self.nAgents)]


        
        self.ch_rate_print = [[i, 1] for i in self.vPort_id]
        self.distance_print = [[(i, j), self.distance[i][j]] for i in self.vPort_id for j in self.vPort_id ]
        # self.segment_distance_print = [[(i, j), self.segment_distance[i][j]] for i in self.vPort_id for j in self.vPort_id ]
        self.edge_print = [(i, j) for i in self.vPort_id for j in self.vPort_id if i != j]
        # self.edge_print = [[k] + i for k, i in zip(range(0, len(self.edge_print)), self.edge_print)] 
        
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
        
        # f.write('%segment distance = distance/5.\n')    
        # for i in self.segment_distance_print:
            # f.write('segment_distance' + str(tuple(i)).replace("'","") + '.\n')        
                             
        #edge
        # for i in self.edge_print:
        #     f.write('edge' + str(tuple(i)).replace("'","") + '.\n')              
        for i in self.vPort_id:
            for j in self.vPort_id:
                f.write('edge' + '(' + i + ',' + j + ').\n')   
        f.close()
    
            

# def GenerateRandomMap(nAgent, nVertices, nEdge, fName):
if __name__ == '__main__':
    if len(sys.argv) > 1:
        nbDrones = sys.argv[1]
        nbOrder = sys.argv[2]
        a = GenerateMapGivenPath(int(nbDrones), int(nbOrder))
        a.run()
    else:
        a = GenerateMapGivenPath()
        a.run()