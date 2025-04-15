% % ------ optimization_3: Maximize revenue
revenue(D, S, W*S) :- as(D, E, W, S), distance(E,S).
:~ revenue(D, S, V). [-V@3, D, S]