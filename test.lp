
% Define nodes
node(1..4).

% Define available colors
color(1..3).

% Define edges (you can modify these as needed)
edge(1,2). edge(1,3). edge(2,3). edge(2,4). edge(3,4).

% Each node must be assigned exactly one color
{ assign_color(N, C) : color(C) } = 1 :- node(N).

% No two connected nodes can have the same color
:- edge(X, Y), assign_color(X, C), assign_color(Y, C).

% Display the result
#show assign_color/2.