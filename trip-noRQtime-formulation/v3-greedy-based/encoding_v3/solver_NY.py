import clingo
import time

# Define the ASP program as a string
asp_program = """
% Example ASP program
as(D, (V, V'), X) :- as_w(D, (V, V'), X, W).
as_w(D, (V, V'), X, W) :- as(D, (V, V'), X).

% Ensure that if as(D, (V, V'), X) is true, as_w(D, (V, V'), X, W) must also be true
:- as(D, (V, V'), X), not as_w(D, (V, V'), X, W).

% Ensure that if as_w(D, (V, V'), X, W) is true, as(D, (V, V'), X) must also be true
:- as_w(D, (V, V'), X, W), not as(D, (V, V'), X).

% Example facts
as(1, (2, 3), 4).
as_w(1, (2, 3), 4, 5).

#show as/3.
#show as_w/4.
"""

# Create a Clingo control object
ctl = clingo.Control()

# Add the ASP program to the control object
ctl.add("base", [], asp_program)

# Ground the program
ctl.ground([("base", [])])

# Define a function to handle each model (answer set)
def on_model(model):
    global start_time
    end_time = time.time()
    solving_time = end_time - start_time
    print(f"Solving time for this answer set: {solving_time:.4f} seconds")
    print(f"Answer set: {model}")

# Measure the time taken to compute each answer set
start_time = time.time()
ctl.solve(on_model=on_model)