import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

data = [np.fromfile("data/"+fname, dtype=np.dtype('float64')) for fname in os.listdir("data")]
data = np.concatenate(data)
print(data.shape)
#data = data.reshape(50,-1)
data = data.reshape(-1,1500)
print(data.shape)
print(data)

# the minimal amount of variance explained
n_components = 0.99

train_data = data
test_data = train_data

# Standardize the dataset (mean = 0, variance = 1)
# Necessary for the PCA to be useful
scaler = StandardScaler()
scaler.fit(train_data)


train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

pca = PCA(n_components, svd_solver="full")

x2 = pca.fit_transform(train_data)
print(pca.explained_variance_ratio_)
print(x2.shape)
