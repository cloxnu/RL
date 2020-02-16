import numpy as np

print(np.random.choice(np.array(range(2)), size=(4, 32), p=[1 - 0.9, 0.9]))