import statsmodels.api as sm
import numpy as np

y = np.random.randn(100)
u = np.random.randn(100, 2)
armax = sm.tsa.ARMA(y, order=(3, 3), exog=u).fit()
