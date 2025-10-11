import sympy as sp

def F_Mat():
    # 1. Define the variables as SymPy symbols
    Lx, Ly, Rx, Ry, A = sp.symbols('Lx Ly Rx Ry A')

    # 2. Define the function F
    numerator = Lx - Rx
    denominator = Ry - Ly

    theta = sp.atan2(numerator, denominator)

    rotZ = sp.Matrix([
        [sp.cos(theta), -sp.sin(theta), 0],
        [sp.sin(theta),  sp.cos(theta), 0],
        [0,           0,          1]
    ])

    A_vec = sp.Matrix([-A, 0, 0])

    F = sp.Matrix([
        [(Lx+Rx)/2 + (rotZ*A_vec)[0]],
        [(Ly+Ry)/2 + (rotZ*A_vec)[1]],
        [theta],
    ])

    return F

def End_Pose(data):
    Lx, Ly, Rx, Ry, A = sp.symbols('Lx Ly Rx Ry A')
    variables = [Lx, Ly, Rx, Ry, A]
    F_callable = sp.lambdify(variables, F_Mat(), "numpy")

    return F_callable(data[0], data[1], data[2], data[3], data[4])


def Cov(data):
    Lx, Ly, Rx, Ry, A = sp.symbols('Lx Ly Rx Ry A')
    variables = [Lx, Ly, Rx, Ry, A]

    J = F_Mat().jacobian(variables)

    C_x = sp.Matrix([
        [4.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 4.0]
    ])

    C_F = J*C_x*sp.transpose(J)

    C_F_callable = sp.lambdify(variables, C_F, "numpy")

    return C_F_callable(data[0], data[1], data[2], data[3], data[4])
