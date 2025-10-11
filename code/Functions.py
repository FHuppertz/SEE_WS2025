import sympy as sp

def F_Mat():
    # 1. Define the variables as SymPy symbols
    Lx, Ly, Rx, Ry, A, T = sp.symbols('Lx Ly Rx Ry A T')

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
        [(Lx+Rx)/2 + A + (rotZ*A_vec)[0]],
        [(Ly+Ry)/2 - T + (rotZ*A_vec)[1]],
        [theta],
    ])


    return F

def End_Pose(data):
    Lx, Ly, Rx, Ry, A, T = sp.symbols('Lx Ly Rx Ry A T')
    variables = [Lx, Ly, Rx, Ry, A, T]
    F_callable = sp.lambdify(variables, F_Mat(), "numpy")

    end_pose = F_callable(data['Lx'], data['Ly'], data['Rx'], data['Ry'], 5.0, 4.75).flatten()

    #print(end_pose[2])

    return end_pose


def Cov(data):
    Lx, Ly, Rx, Ry, A, T = sp.symbols('Lx Ly Rx Ry A T')
    variables = [Lx, Ly, Rx, Ry, A, T]

    J = F_Mat().jacobian(variables)

    C_x = sp.Matrix([
        [4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 4.0]
    ])

    C_F = J*C_x*sp.transpose(J)

    C_F_callable = sp.lambdify(variables, C_F, "numpy")

    return C_F_callable(data['Lx'], data['Ly'], data['Rx'], data['Ry'], 10.0, 4.75).flatten()
