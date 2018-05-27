from __future__ import division
import numpy as np


# gmm_EM now with numpy :)
class gmm_EM:
    def __init__(self, nb_clust, dim, centers=None, covars=None, weights=None):

        self.nb_clust = nb_clust
        self.dim = dim

        self.covar_eps = 1e-4

        #         torch.manual_seed(0)

        if centers != None:
            self.centers = centers
        else:
            self.centers = 1 * np.random.randn(self.nb_clust, self.dim)

        if covars != None:
            self.covars = covars
        else:
            self.covars = np.ones((self.nb_clust, self.dim))

        if weights != None:
            self.weights = weights
        else:
            self.weights = (1 / self.nb_clust) * np.ones(self.nb_clust)

    def get_log_likelihoods(self, data, eps=None):

        eps = self.covar_eps

        # Returns likelihoods of each datapoint for every cluster
        if (self.covars < eps).any():
            self.covars += (self.covars < eps).astype(int) * eps

        # Nclust
        log_det_term = -0.5 * (np.log(self.covars).sum(axis=1))
        # 1, Nclust
        norm_term = np.expand_dims(log_det_term, axis=0) - 0.5 * np.log(2 * np.pi) * self.dim
        # 1, Nclusters, dims
        inv_covars = np.expand_dims(1 / self.covars, axis=0)
        # batch_size, Nclust, dims
        dist = (np.expand_dims(self.centers, axis=0) - np.expand_dims(data, axis=1)) ** 2
        # batch_size, Ncenters
        exponent = (-0.5 * dist * inv_covars).sum(axis=2)
        # batch_size, Ncenters
        log_p = norm_term + exponent
        return log_p

    def get_log_likelihood(self, data_point, covar_eps=None):

        covar_eps = self.covar_eps

        # Returns likelihoods of each datapoint for every cluster
        if (self.covars < covar_eps).any():
            self.covars += (self.covars < covar_eps).astype(int) * covar_eps

        # Nclust
        log_det_term = -0.5 * (self.covars.log().sum(axis=1))
        # Nclust
        norm_term = log_det_term - 0.5 * np.log(2 * np.pi) * self.dim
        # Nclusters, dims
        inv_covars = (1 / self.covars)
        # Nclust, dims
        dist = (self.centers - np.expand_dims(data_point, axis=0)) ** 2
        # Ncenters
        exponent = (-0.5 * dist * inv_covars).sum(axis=1)
        # batch_size, Ncenters
        log_p = norm_term + exponent
        return log_p

    def E_step(self, data, eps=1e-8):
        # get_posteriors of clusters for each datapoint-> responsabilities
        # batch_size, Ncenters
        log_p = self.get_log_likelihoods(data)
        #         print('loglikelihoods', log_p)
        log_p -= log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        #         print('likelihoods', p)
        # 1, Ncenters
        expand_weights = np.expand_dims(self.weights, axis=0)
        # batch_size, 1
        p_sum = (p * expand_weights).sum(axis=1, keepdims=True)
        # batch_size, Ncenters
        responsabilities = p * expand_weights / (p_sum + eps)
        return responsabilities

    def M_step(self, data, responsabilities, eps=1e-6):
        # responsabilities: batch_size, Ncenters
        # 1
        Ndat = data.shape[0]
        # Ncenters
        N_k = np.sum(responsabilities, axis=0) + eps
        self.weights = N_k / (Ndat + eps * len(N_k))
        # batch_size, Ncenters, dims
        weighed_data = np.expand_dims(data, axis=1) * np.expand_dims(responsabilities, axis=2)
        # Ncenters, dims
        self.centers = weighed_data.sum(axis=0) / np.expand_dims(N_k, axis=1)
        # batch_size, Ncenters, dims
        dist = (np.expand_dims(self.centers, axis=0) - np.expand_dims(data, axis=1)) ** 2
        # Ncenters, dims
        print('N_k', N_k)
        self.covars = (np.expand_dims(responsabilities, axis=2) * dist).sum(axis=0) / np.expand_dims(N_k, axis=1)

        # Returns likelihoods of each datapoint for every cluster
        if (self.covars < self.covar_eps).any():
            self.covars += (self.covars < self.covar_eps).astype(int) * self.covar_eps

        return None

    def update_params(self, data, EM_iters=1):

        E = np.zeros(EM_iters + 1)
        #       E[0] = self.get_cost(data)
        for i in range(EM_iters):
            resp = self.E_step(data)
            self.M_step(data, resp)

        #       print('centers', self.centers)
        #       print('covars', self.covars)
        print('weights', self.weights)
        #           E[i+1] = gmm.get_cost(data)

        # return E

    def get_cost(self, data, eps=1e-100):
        # Returns -loglike of all data

        if np.isscalar(data) or len(data.shape) == 1:
            # Ncenters
            p = np.exp(self.get_log_likelihood(data))
            # Ncenters
            expand_weights = self.weights
            # 1
            p_sum = (p * expand_weights).sum()
            # 1
            E = -np.log(p_sum)
        else:
            # batch_size, Ncenters
            p = np.exp(self.get_log_likelihoods(data))

            # 1, Ncenters
            expand_weights = np.expand_dims(self.weights, axis=0)
            # batch_size, 1
            p_sum = (p * expand_weights).sum(axis=1, keepdims=False)

            if (p_sum < eps).any():
                p_sum += (p_sum < eps).astype(int) * eps
            # batch_size
            E = -np.log(p_sum)
        return E

class onedim_Gaussian(object):

    def __init__(self, mu, var):
        self.mu = mu
        self.var = var
        self.get_cost = self.get_minus_log_likelihood

    def get_minus_log_likelihood(self, X, eps=1e-6):
        # Returns likelihoods of each datapoint for every cluster
        if self.var < eps:
            self.var = eps
        # 1
        log_det_term = -0.5 * np.log(self.var)
        # 1
        norm_term = -0.5 * np.log(2 * np.pi)
        # batch_size
        dist = (X - self.mu) ** 2
        # batch_size
        exponent = (-0.5 * dist / self.var)
        # print('exponent shape:', exponent.shape)
        # batch_size
        log_p = log_det_term + exponent + norm_term
        # print('log_p shape:', log_p.shape)
        return -log_p

    def update_params(self, X):
        self.mu = np.mean(X)
        self.var = np.var(X)
