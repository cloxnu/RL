import numpy as np
import pandas as pd

actions = ['1', '2', '3', '4']

l = pd.DataFrame(
    [[2, 2], [4, 5], [7, 8]],
    index=['cobra', 'viper', 'sidewinder'],
    columns=['max_speed', 'shield']
)

s = l.loc['cobra', :]
print(s[s == np.max(s)].index)