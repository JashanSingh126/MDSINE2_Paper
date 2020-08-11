import numpy as np
import pylab as pl
import logging
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import sys
logging.basicConfig(level=logging.INFO)

# Load the dataset
dset = load_iris()
X = dset.data
y = dset.target
n_features = X.shape[1]

# Make the graph and define the data
G = pl.Graph(name='g1')
data = pl.Data(X=X, y=y, G=G)

# Set the process variance and the prior for b
pv = pl.Variable(value=1, name='pv', G=G)

class PriorVariance(pl.variables.SICS):
    def initialize(self):
        # Use least squares solution of coefficients to set the prior
        # to set mean of the prior
        self.prior.dof.value = 2.5

        resid = pl.inference.OLS(X,y).reshape(-1,1)
        se = np.sum(np.square(resid.ravel()))

        self.prior.scale.value = se * (self.prior.dof.value - 2) / \
            self.prior.dof.value
        self.value = self.prior.mean()

    def update(self):
        resid = self.G['b'].value.reshape(-1,1)
        se = np.sum(np.square(resid.ravel()))
        n = len(resid)

        self.dof.value = self.prior.dof.value + n
        self.scale.value = ((self.prior.scale.value * self.prior.dof.value) + \
           se)/self.dof.value
        self.sample()

# Define the posterior for the coefficients
class Coefficients(pl.variables.MVN):
    def initialize(self):
        pass

    def update(self):
        X = self.G.data.X
        y = self.G.data.y
        pv = self.G['pv'].value

        prior_mean = self.prior.mean.value
        prior_var = self.prior.var.value

        prec = X.T @ X / pv + (1/prior_var)
        cov = np.linalg.pinv(prec)

        self.mean.value = (cov @ X.T @ y / pv) + (prior_mean/prior_var)
        self.cov.value = cov
        self.sample()

# Initialize b and set the prior
varprior = pl.variables.SICS(G=G)
priorvar = PriorVariance(name='b_prior_var', G=G)
priorvar.add_prior(varprior)

bprior = pl.variables.Normal(mean=0, var=priorvar, name='b_prior', G=G)
b = Coefficients(G=G, name='b', shape=(n_features,))
b.add_prior(bprior)

# Begin inference
mcmc = pl.inference.BaseMCMC(burnin=500, n_samples=1000, graph=G)
mcmc.set_inference_order(order=['b', 'b_prior_var'])

# Initialize
priorvar.initialize()

mcmc.run(log_every=5)

a = pl.inference.OLS(covariates=X, observations=y)
for i in range(n_features):
    axleft, _ = pl.visualization.render_trace(var=b, idx=i)
    axleft.axvline(x=a[i], color='red', label='OLS solution')
    axleft.legend()
    fig = plt.gcf()
    fig.suptitle('idx {}'.format(i))
pl.visualization.render_trace(var=priorvar)
fig = plt.gcf()
fig.suptitle('Variance b')
plt.show()


