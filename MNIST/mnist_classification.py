import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from mnist import MNIST

from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import time

mnist = MNIST('data')
x_training, y_training = mnist.load_training()

#print(x_training[:20])
x_training = np.asarray(x_training).astype(np.float32)
plt.figure()
plt.imshow(x_training[0].reshape(28,28))
plt.show()




