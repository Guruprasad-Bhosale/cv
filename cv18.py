import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
X = np.arange(100).reshape(-1,1)
y = 2 * X.squeeze() + 10
y[20:30] += 100
ransac = RANSACRegressor()
ransac.fit(X, y)
line_y = ransac.predict(X)
plt.scatter(X, y)
plt.plot(X, line_y)
plt.show()