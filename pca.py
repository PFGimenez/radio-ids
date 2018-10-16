import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# the minimal amount of variance explained
n_components = 0.95

train_data = np.random.random(1000000).reshape(10000,100)
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
