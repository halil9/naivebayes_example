import numpy as np

x = np.array([[-2, 4], [6, 8], [1, 3], [-4, 7], [1, 5], [-2, 2], [2, 2], [4, 5], [5, 0], [3, 6]])
y = np.array([4, 4, 4, 4, 5, 5, 4, 5, 5, 4])
from sklearn.naive_bayes import GaussianNB

module = GaussianNB()
module.fit(x, y)
tahmin = module.predict([[-4, 7], [1, 5]])
print(module.predict([[-4, 7], [1, 5]]))
