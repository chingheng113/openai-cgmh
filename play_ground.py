import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


top_1 = pd.read_csv('1.0000001192092896.csv')
top_2 = pd.read_csv('0.9912627935409546.csv')
top_3 = pd.read_csv('0.8041002750396729.csv')
top_4 = pd.read_csv('0.7899750471115112.csv')
top_1.plot.hist()
print(np.mean(top_1.values), np.std(top_1.values))
top_2.plot.hist()
print(np.mean(top_2.values), np.std(top_2.values))
top_3.plot.hist()
print(np.mean(top_3.values), np.std(top_3.values))
top_4.plot.hist()
print(np.mean(top_4.values), np.std(top_4.values))

plt.show()