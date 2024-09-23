
vertiport(lbm;mgt;tci;mbf;crj).
% vertiport(lbm;mgt;tci).
%LMP = 0.12$/kWh; MER = 0.000329 tons/kWh, gamma=185$/tons; lmp_mer = discharge_rate*(LMP+gamma*MER) = 4*0.19 = 0.76
%average lmp_mer = 0.76
%range for lmp_mer : [1 - 8]
%multiply by 2.

% 1 MW
ch_rate(lbm, 1).
ch_rate(mgt, 1).
ch_rate(tci, 1).
ch_rate(mbf, 1).
ch_rate(crj, 1).
% distance= miles
distance((lbm, lbm), 0).
distance((lbm, mgt), 130).
distance((lbm, tci), 220).
distance((lbm, mbf), 90).
distance((lbm, crj), 110).
distance((mgt, lbm), 130).
distance((mgt, mgt), 0).
distance((mgt, tci), 100).
distance((mgt, mbf), 100).
distance((mgt, crj), 60).
distance((tci, lbm), 220).
distance((tci, mgt), 100).
distance((tci, tci), 0).
distance((tci, mbf), 200).
distance((tci, crj), 160).
distance((mbf, lbm), 90).
distance((mbf, mgt), 100).
distance((mbf, tci), 200).
distance((mbf, mbf), 0).
distance((mbf, crj), 40).
distance((crj, lbm), 110).
distance((crj, mgt), 60).
distance((crj, tci), 160).
distance((crj, mbf), 40).
distance((crj, crj), 0).
%segment distance = distance/5.
%total revenue = 960
segment_distance((lbm, lbm), 0).
segment_distance((lbm, mgt), 8).
segment_distance((lbm, tci), 4).
segment_distance((lbm, mbf), 2).
segment_distance((lbm, crj), 2).
segment_distance((mgt, lbm), 8).
segment_distance((mgt, mgt), 0).
segment_distance((mgt, tci), 2).
segment_distance((mgt, mbf), 2).
segment_distance((mgt, crj), 1).
segment_distance((tci, lbm), 4).
segment_distance((tci, mgt), 2).
segment_distance((tci, tci), 0).
segment_distance((tci, mbf), 4).
segment_distance((tci, crj), 8).
segment_distance((mbf, lbm), 2).
segment_distance((mbf, mgt), 2).
segment_distance((mbf, tci), 4).
segment_distance((mbf, mbf), 0).
segment_distance((mbf, crj), 1).
segment_distance((crj, lbm), 2).
segment_distance((crj, mgt), 1).
segment_distance((crj, tci), 8).
segment_distance((crj, mbf), 1).
segment_distance((crj, crj), 0).

edge(lbm, mgt).
edge(lbm, tci).
edge(lbm, mbf).
edge(lbm, crj).
edge(mgt, lbm).
edge(mgt, tci).
edge(mgt, mbf).
edge(mgt, crj).
edge(tci, lbm).
edge(tci, mgt).
edge(tci, mbf).
edge(tci, crj).
edge(mbf, lbm).
edge(mbf, mgt).
edge(mbf, tci).
edge(mbf, crj).
edge(crj, lbm).
edge(crj, mgt).
edge(crj, tci).
edge(crj, mbf).

edge(crj, crj).
edge(mbf, mbf).
edge(tci, tci).
edge(lbm, lbm).
edge(mgt, mgt).

