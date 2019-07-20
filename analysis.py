import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

X = pd.read_csv('train_values.csv')
y = pd.read_csv('train_labels.csv')
X = X.merge(y, on = 'building_id')

sns.swarmplot(x = 'damage_grade', y = 'age', data = X)
plt.show()