% Graph definition
node(a; b; c; d).
edge(a,b). edge(b,c). edge(c,d). edge(d,a). edge(a,c).

% Each node can be in the cover or not
{ cover(X) } :- node(X).

% Every edge must be covered by at least one endpoint
:- edge(X,Y), not cover(X), not cover(Y).

% Assign weights (costs) to nodes
weight(a, 2).
weight(b, 1).
weight(c, 3).
weight(d, 1).

% Minimize the total weight of selected nodes
#minimize { W,X : cover(X), weight(X,W) }.