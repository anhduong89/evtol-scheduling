% agent(0..9).
% init_loc(0, mbf).
% init_loc(1, mgt).
% init_loc(2, crj).
% init_loc(3, lbm).
% init_loc(4, mgt).
% init_loc(5, mgt).
% init_loc(6, crj).
% init_loc(7, mgt).
% init_loc(8, lbm).
% init_loc(9, mgt).
% %capacity 4 of agent 0
% capacity(0, 4).
% capacity(1, 4).
% capacity(2, 4).
% capacity(3, 4).
% capacity(4, 4).
% capacity(5, 4).
% capacity(6, 4).
% capacity(7, 4).
% capacity(8, 4).
% capacity(9, 4).
% %energy capacity max = 100 kWh of agent 0
% emax(0, 100).
% emax(1, 100).
% emax(2, 100).
% emax(3, 100).
% emax(4, 100).
% emax(5, 100).
% emax(6, 100).
% emax(7, 100).
% emax(8, 100).
% emax(9, 100).
% %energy capacity min = 0 kWh for agent 0emin(0, 0).
% emin(1, 0).
% emin(2, 0).
% emin(3, 0).
% emin(4, 0).
% emin(5, 0).
% emin(6, 0).
% emin(7, 0).
% emin(8, 0).
% emin(9, 0).
% %discharge rate = 4kWh/miles for agent 0dischg_rate(0, 4).
% dischg_rate(1, 4).
% dischg_rate(2, 4).
% dischg_rate(3, 4).
% dischg_rate(4, 4).
% dischg_rate(5, 4).
% dischg_rate(6, 4).
% dischg_rate(7, 4).
% dischg_rate(8, 4).
% dischg_rate(9, 4).
%request(ID, (edge), number of request passenger)request(0, (mgt, lbm), 20).
request(1, (tci, lbm), 20).
request(2, (mbf, lbm), 20).
request(3, (crj, lbm), 20).
request(4, (lbm, mgt), 20).
request(5, (tci, mgt), 20).
request(6, (mbf, mgt), 20).
request(7, (crj, mgt), 20).
request(8, (lbm, tci), 20).
request(9, (mgt, tci), 20).
request(10, (mbf, tci), 20).
request(11, (crj, tci), 20).
request(12, (lbm, mbf), 20).
request(13, (mgt, mbf), 20).
request(14, (tci, mbf), 20).
request(15, (crj, mbf), 20).
request(16, (lbm, crj), 20).
request(17, (mgt, crj), 20).
request(18, (tci, crj), 20).
request(19, (mbf, crj), 20).
vertiport(lbm;mgt;tci;mbf;crj).
%LMP = 0.12$/kWh; MER = 0.000329 tons/kWh, gamma=185$/tons; lmp_mer = discharge_rate*(LMP+gamma*MER) = 4*0.19 = 0.76
%average lmp_mer = 0.76
%range for lmp_mer : [1 - 3]
%multiply by 2.
lmp_mer(lbm, 0, 2).
lmp_mer(mgt, 0, 1).
lmp_mer(tci, 0, 2).
lmp_mer(mbf, 0, 1).
lmp_mer(crj, 0, 2).
lmp_mer(lbm, 1, 1).
lmp_mer(mgt, 1, 2).
lmp_mer(tci, 1, 1).
lmp_mer(mbf, 1, 2).
lmp_mer(crj, 1, 1).
lmp_mer(lbm, 2, 1).
lmp_mer(mgt, 2, 2).
lmp_mer(tci, 2, 2).
lmp_mer(mbf, 2, 1).
lmp_mer(crj, 2, 1).
lmp_mer(lbm, 3, 2).
lmp_mer(mgt, 3, 1).
lmp_mer(tci, 3, 2).
lmp_mer(mbf, 3, 2).
lmp_mer(crj, 3, 2).
lmp_mer(lbm, 4, 2).
lmp_mer(mgt, 4, 1).
lmp_mer(tci, 4, 1).
lmp_mer(mbf, 4, 1).
lmp_mer(crj, 4, 1).
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
segment_distance((lbm, mgt), 3).
segment_distance((lbm, tci), 4).
segment_distance((lbm, mbf), 2).
segment_distance((lbm, crj), 2).
segment_distance((mgt, lbm), 3).
segment_distance((mgt, mgt), 0).
segment_distance((mgt, tci), 2).
segment_distance((mgt, mbf), 2).
segment_distance((mgt, crj), 1).
segment_distance((tci, lbm), 4).
segment_distance((tci, mgt), 2).
segment_distance((tci, tci), 0).
segment_distance((tci, mbf), 4).
segment_distance((tci, crj), 3).
segment_distance((mbf, lbm), 2).
segment_distance((mbf, mgt), 2).
segment_distance((mbf, tci), 4).
segment_distance((mbf, mbf), 0).
segment_distance((mbf, crj), 1).
segment_distance((crj, lbm), 2).
segment_distance((crj, mgt), 1).
segment_distance((crj, tci), 3).
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

null_edge((mgt, mgt)).
null_edge((lbm, lbm)).
null_edge((tci, tci)).
null_edge((mbf, mbf)).
null_edge((crj, crj)).