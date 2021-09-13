import numpy as np
from scipy import stats

n = 10
a = np.random.randn(n) + 0.1
b = np.random.randn(n) + 2

t2, p2 = stats.ttest_ind(a, b)
print('t', str(t2), 'p', str(p2))