"""
    Hidden Markov Model learning
"""

import numpy as np
from hmmlearn import hmm

nb_states = 5

x = np.random.random(1000).reshape(-1,5)
print(x)
model = hmm.GaussianHMM(nb_states, "full")
model.fit(X)
z = model.predict(X)
print(z)

# evaluation : model.score(X) / model.score(X[:-1])

