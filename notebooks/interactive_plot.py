import pandas as pd
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize
from sklearn.decomposition import PCA, KernelPCA
import instructions

path = os.path.split(os.getcwd())
#data_directory = os.path.join(path[0], 'data\\raw')
data_directory = os.path.join(os.getcwd(), 'data\\raw')


X = instructions.load_csv(data_directory, 'train_data')
y = instructions.load_csv(data_directory, 'train_labels')

y_binar = binarize(y)
y = pd.DataFrame(np.ravel(y_binar))

kpca = KernelPCA(n_components=3, kernel="linear", gamma=0.5)

X_kpca = kpca.fit_transform(X)


fig = plt.figure()
ax = Axes3D(fig)

ax.scatter3D(X_kpca[:,1],X_kpca[:,2], X_kpca[:,0], c=y[0])

plt.show()