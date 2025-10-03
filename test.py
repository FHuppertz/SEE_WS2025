import sympy as sp
import numpy as np

# 1. Define the variables as SymPy symbols
x11, x12, x21, x22 = sp.symbols('x11 x12 x21 x22')

# 2. Define the function F
# F = arcsin((x-y) / sqrt((x-y)^2 + (z+a)^2))
numerator = x21 - x11
denominator = x22 - x12

F = sp.atan(numerator / denominator)

# 3. Calculate the partial derivatives (for the Jacobian)
# Since F is a scalar function (a single output), its Jacobian is just a row vector
# of partial derivatives (the gradient vector).

dF_dx11 = sp.diff(F, x11)
dF_dx12 = sp.diff(F, x12)
dF_dx21 = sp.diff(F, x21)
dF_dx22 = sp.diff(F, x22)

# 4. Print the results
print("Function F:")
sp.pprint(sp.simplify(F))
print("-" * 30)

print("Partial Derivative dF/dx:")
sp.pprint(sp.simplify(dF_dx11))
print("-" * 30)

print("Partial Derivative dF/dy:")
sp.pprint(sp.simplify(dF_dx12))
print("-" * 30)

print("Partial Derivative dF/dz:")
sp.pprint(sp.simplify(dF_dx21))
print("-" * 30)

print("Partial Derivative dF/da:")
sp.pprint(sp.simplify(dF_dx22))
