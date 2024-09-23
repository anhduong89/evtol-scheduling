

% % revenue if agents fly in the arc (V, V') at time T can serve all passengers of request of arc(V, V') at time T
% revenue(N, WR) :- edge_w((V, V'), W, T), request(N, V, V', WR, T), W > WR.
% % revenue if agents fly in the arc (V, V') at time T can't serve all passengers of request of arc(V, V') at time T
% revenue(N, W) :- edge_w((V, V'), W, T), request(N, V, V', WR, T), W <= WR.


% revenue if agents fly in the arc (V, V') at time T can serve all passengers of request of arc(V, V') at time T
revenue(N, R/2) :- edge_w((V, V'), W, T), request(N, V, V', WR, T), W > WR, distance((V, V'), D), R = WR * D * alpha.
% revenue if agents fly in the arc (V, V') at time T can't serve all passengers of request of arc(V, V') at time T
revenue(N, R/2) :- edge_w((V, V'), W, T), request(N, V, V', WR, T), W <= WR, distance((V, V'), D), R = W * D * alpha.
% find edge_w that maximize served passengers
#maximize{R,N:revenue(N, R)}.

% #minimize{C@0, D, X:emission_cost(C, D, X)}.
% #minimize{C@0, D, X:charging_cost(C, D, X)}.

passenger(N, WR) :- edge_w((V, V'), W, T), request(N, V, V', WR, T), W > WR.
passenger(N, W) :- edge_w((V, V'), W, T), request(N, V, V', WR, T), W <= WR.
total_passengers(T) :- T = #sum{W,N:passenger(N, W)}.


% #show edge_w/3.
#show total_passengers/1.
