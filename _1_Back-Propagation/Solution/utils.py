# _1_Back-Propagation/Solution/neuron.py

# %%
import warnings; warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np

from typing import Any, Tuple

# %%
np.random.seed(0)

# %%
def create_data(n: int, k: int, d: int = 2) -> Tuple[np.ndarray, np.ndarray]:
  X: np.ndarray = np.zeros((n*k, d))
  y: np.ndarray = np.zeros(n*k, dtype='uint8')

  for i in range(k):
    ix: range = range(n * i, n * (i + 1))
    r: Tuple[Any, Any | float] | Any = np.linspace(0.0, 1.0, n)
    t: Any = np.linspace(i * 4, (i + 1) * 4, n) + np.random.randn(n) * 0.8
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = i

  return X, y


def plot_data(X: np.ndarray, y: np.ndarray) -> None:
	plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
	plt.show()

# %%
if __name__ == '__main__':
  X: np.ndarray; y: np.ndarray
  X, y = create_data(100, 3)
  plot_data(X, y)
