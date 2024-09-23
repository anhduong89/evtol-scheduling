1 {agent_move(D, (V, V'), T): edge_w((V, V'), EW, T), EW > 0; agent_move(D, (V, V), T)} 1 :- agent_loc(D, V, T).

agent_loc(D, V', T+S) :- agent_move(D, (V, V'), T), distance_t((V, V'), S).

% :- edge_w((V, V'), W, T), C = #count{D: agent_move(D, (V, V'), T)}, C!=W.

agent_loc(0,mgt,0).agent_loc(1,crj,0).agent_loc(2,tci,0).agent_loc(3,mgt,0).agent_loc(4,mbf,0).edge_w((crj,mbf),4,0).total_passengers(4).edge_w((mgt,mgt),4,0).edge_w((mgt,mbf),4,0).edge_w((tci,crj),4,0).edge_w((mbf,mbf),4,0).edge_w((crj,lbm),4,4).edge_w((mbf,lbm),4,4).edge_w((mbf,mgt),4,2).edge_w((mbf,lbm),4,2).edge_w((mbf,mbf),4,1).edge_w((mgt,mbf),4,6).edge_w((mgt,lbm),4,1).edge_w((lbm,mgt),4,6).edge_w((lbm,crj),4,5).

distance_t((V, V'), ((S/(speed*30))+1)*2) :- distance((V, V'), S), V != V'.
distance_t((V, V), 1) :- edge(V, V).%horizon in minutes

#show agent_move/3.
#show agent_loc/3.