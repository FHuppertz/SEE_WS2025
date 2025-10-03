import sympy as sp

# 1. Define the variables as SymPy symbols
x, y, z, a = sp.symbols('x y z a')

# 2. Define the function F
# F = arcsin((x-y) / sqrt((x-y)^2 + (z+a)^2))
numerator = x - y
denominator_squared = (x - y)**2 + (z + a)**2
denominator = sp.sqrt(denominator_squared)

F = sp.asin(numerator / denominator)

# 3. Calculate the partial derivatives (for the Jacobian)
# Since F is a scalar function (a single output), its Jacobian is just a row vector
# of partial derivatives (the gradient vector).

dF_dx = sp.diff(F, x)
dF_dy = sp.diff(F, y)
dF_dz = sp.diff(F, z)
dF_da = sp.diff(F, a)

# 4. Print the results
print("Function F:")
sp.pprint(F)
print("-" * 30)

print("Partial Derivative dF/dx:")
sp.pprint(dF_dx)
print("-" * 30)

print("Partial Derivative dF/dy:")
sp.pprint(dF_dy)
print("-" * 30)

print("Partial Derivative dF/dz:")
sp.pprint(dF_dz)
print("-" * 30)

print("Partial Derivative dF/da:")
sp.pprint(dF_da)
