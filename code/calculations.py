import numpy as np
import pandas as pd

from Functions import End_Pose, Cov

## TEST
# [Lx, Ly, Rx, Ry, A]
data = [10, 0, 0, 0, 10]

print(End_Pose(data))

print(Cov(data))
