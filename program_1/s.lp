% ------ INSTANCES
#include "instances/network_NY.lp".
#include "instances/init.lp".
#include "instances/rq.lp".

max_seg(D,max_seg):-agent(D).
seg(1..max_seg).

% ------ GENERATE ROUTING

% -- assign edge to each segment
1 {as(D,E,0..C,X):edge(E)} 1 :- agent(D), capacity(D,C), X=start_seg..max_seg.

% total delivered customers can not excess demand
:- (R+1) #sum{W,D,X:as(D,E,W,X)}, edge(E), request(_,E,R).

% start vertiport as specicfied
:- init_loc(D,V), as(D,(V',_),_,1), V!=V'.

% consecutive trips need to be connected
:-as(D,(V,V1),_,X),as(D,(V2,V3),_,X+1),V1!=V2.


% ------ GENERATE ROUTING

% at(D, L, 1) :- init_loc(D, L). 

% start((L, L'), L):- edge((L, L')). 

% 1 {at(D, L', X+1) : edge((L,L'))} 1 :- agent(D), capacity(D,C), seg(X), at(D, L, X).

% % -- assign edge to each segment

% % total delivered customers can not excess demand
% :- (R+1) #sum{W,D,X:as(D,E,W,X)}, edge(E), request(_,E,R).

% 1 {as(D,(L,L'),0..C,X)} 1 :- 
%     agent(D), capacity(D,C), seg(X), edge((L,L')), at(D, L, X), at(D, L', X+1), L !=L'.

:- as(D,E,W,_), as(D',E,W',_), W>0, W'>0, D!=D', capacity(D,C), W+W'<C.

:- as(D,E,W,X), as(D,E,W',X'), W>0, W'>W, X'>X, capacity(D,C), W+W'<C.

:~ as(_,_,W,_), W>0. [-W@3]

total_served(E, T) :- edge(E), T = #sum{W,D,X:as(D,E,W,X)}.

:~ total_served(_, T). [-T@2]  

% remain_cust(E,N,0) :- request(_,E,N).
% remain_cust(E,N-P,X) :- remain_cust(E,N,X-1), P=#sum{W,D: as(D,E,W,X)}, seg(X).


#show.
% #show at/3.
#show as/4.
% #show remain_cust/3. 