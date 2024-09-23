import pyomo.environ as pyenv

# Define the model
model = pyenv.ConcreteModel()

# Define the decision variables
model.x1 = pyenv.Var(within=pyenv.NonNegativeIntegers)
model.x2 = pyenv.Var(within=pyenv.NonNegativeIntegers)

# Define the objective function
model.obj = pyenv.Objective(expr=100 * model.x1 + 150 * model.x2, sense=pyenv.maximize)

# Define the constraints
model.constraint1 = pyenv.Constraint(expr=8000 * model.x1 + 4000 * model.x2 <= 40000)
model.constraint2 = pyenv.Constraint(expr=15 * model.x1 + 30 * model.x2 <= 200)

# Solve the problem using GLPK solver with a minimum gap of 0.1
solver = pyenv.SolverFactory('glpk')
solver.options['mipgap'] = 0.9  # Set the MIP gap to 0.1
results = solver.solve(model, tee=True)

# Retrieve the upper bound from the solver results
upper_bound = results.problem.upper_bound

# Print the results
print(f"x1: {pyenv.value(model.x1)}")
print(f"x2: {pyenv.value(model.x2)}")
print(f"Optimal Value of the objective function (Z): {pyenv.value(model.obj)}")
print(f"Upper Bound of the objective function (Z): {upper_bound}")


# This Python code snippet is using the Pyomo library to model and solve an optimization problem.
# Here's a breakdown of what each part of the code is doing:
# import pyomo.environ as pyenv

# # Define the model
# model = pyenv.ConcreteModel()

# # Define the decision variables
# model.x1 = pyenv.Var(within=pyenv.NonNegativeIntegers)
# model.x2 = pyenv.Var(within=pyenv.NonNegativeIntegers)

# # Define the objective function
# model.obj = pyenv.Objective(expr=100 * model.x1 + 150 * model.x2, sense=pyenv.maximize)

# # Define the constraints
# model.constraint1 = pyenv.Constraint(expr=8000 * model.x1 + 4000 * model.x2 <= 40000)
# model.constraint2 = pyenv.Constraint(expr=15 * model.x1 + 30 * model.x2 <= 200)

# # Assign specific values to decision variables
# model.x1.value = 2
# model.x2.value = 5

# # Evaluate the objective function with the given assignment
# upper_bound = pyenv.value(model.obj)

# # Print the results
# print(f"x1: {model.x1.value}")
# print(f"x2: {model.x2.value}")
# print(f"Upper Bound of the objective function (Z): {upper_bound}")