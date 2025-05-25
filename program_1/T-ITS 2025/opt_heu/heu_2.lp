% Prioritize scheduling higher-revenue flights earlier in the trajectory
#heuristic as(_,E,W,X):distance(E,S). [W*S*(max_seg-X+1),true]