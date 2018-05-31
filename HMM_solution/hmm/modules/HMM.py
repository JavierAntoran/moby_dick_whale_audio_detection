from __future__ import division
import numpy as np
from GMM import gmm_EM
import cPickle
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from IPython.core.display import HTML, clear_output

class HMM(object):
    delta = None
    phi = None

    def __init__(self, Q, G=None, p0=None, debug=False):

        self.debug = debug
        self.Q = Q

        # simple forward markov chain with equiprobable transitions
        self.A = np.zeros((Q, Q))
        self.A[(np.arange(0, Q - 1), np.arange(0, Q - 1))] = 0.5
        self.A[(np.arange(0, Q - 1), np.arange(1, Q))] = 0.5
        self.A[-1, -1] = 1

        # Prior = starting A
        self.A0 = self.A
        self.A0_sum = np.sum(self.A0, axis=1, keepdims=True)
        self.dprior_weight = 1

        if p0 is None:
            self.p0 = np.zeros(self.Q)
            self.p0[0] = 1
        else:
            self.p0 = p0

        if G is None:
            self.G = np.empty(self.Q, dtype=object)
            for i in range(self.Q):
                self.G[i] = gmm_EM(nb_clust=2, dim=2)
        else:
            self.G = G

    def set_likelihood_mtx(self, x):

        self.B = np.empty((self.Q, x.shape[0]))
        # x: (T windows, N features)
        for i in range(self.Q):
            self.B[i, :] = -1 * self.G[i].get_cost(x)

    def backtracking(self, phi, delta):

        start = np.argmax(phi[:, self.T - 1])
        posV = np.array([start])

        t = self.T - 1
        nextV = start
        while t > 0:
            nextV = delta[int(nextV), int(t)]

            posV = np.append(posV, nextV)
            t -= 1

        return posV

    def update_model(self, trace_vec, x):

        cat_trace = np.concatenate(trace_vec)
        cat_x = np.concatenate(x, axis=0)

        for trace in trace_vec:
            temp_T = trace.shape[0]
            new_A = np.zeros((self.Q, self.Q))

            for p in np.arange(1, temp_T):
                new_A[int(trace[p - 1]), int(trace[p])] += 1

        new_A_sum = np.sum(new_A, axis=1, keepdims=True)
        self.A = (self.dprior_weight * self.A0 + new_A) / (self.dprior_weight * self.A0_sum + new_A_sum)

        print('updated A matrix:', self.A)

        for q in range(self.Q):
            if np.any(cat_trace == q):
                self.G[q].update_params(cat_x[cat_trace == q])

    def train(self, train_seq, iterations=10, N_only_gmm=2):

        self.mean_loglike = np.zeros(iterations)
        for i in range(iterations):

            kk = 0
            traces = []

            print('Starting Iteration %d of %d' % (i, iterations))
            tic0 = time.time()

            for x in train_seq:

                self.T = x.shape[0]

                assert self.T >= self.Q

                if kk % 100 == 0:
                    print('series %d of %d in Iteration %d of %d' % (kk, len(train_seq), i, iterations))
                    plt.close('all')
                    plt.clf()
                    #  clear_output()
                    pass

                if i >= N_only_gmm:
                    self.set_likelihood_mtx(x)
                    trace, min_loglike = self.viterbi(x)

                    self.mean_loglike[i] += min_loglike / len(train_seq)

                    if self.debug:
                        plt.figure(dpi=40)
                        plt.imshow(self.phi, cmap='jet', aspect='auto')
                        plt.plot(np.arange(self.T), trace, 'r--')
                        plt.gca().invert_yaxis()
                        plt.xlabel('TARGET')
                        plt.ylabel('SOURCE')
                        plt.title('HiddenMeMe viterbi')

                else:
                    trace = self.generate_random_uniform_trace()

                    if self.debug:
                        print('generated trace:', trace)
                        phi = np.ones((self.Q, self.T)) * 255
                        plt.figure(dpi=40)
                        plt.imshow(phi, cmap='jet', aspect='auto')
                        plt.plot(np.arange(self.T), trace, 'r--')
                        plt.gca().invert_yaxis()
                        plt.xlabel('TARGET')
                        plt.ylabel('SOURCE')
                        plt.title('HiddenMeMe generated trace')

                kk += 1
                traces.append(trace)

            self.update_model(traces, train_seq)
            tic1 = time.time()
            print('mean loglike: %f' % self.mean_loglike[i])
            print('ellapsed time: %f seconds' % (tic1-tic0))

    def generate_random_uniform_trace(self):
        trace = np.empty(self.T)
        # starting point according to p0
        trace[0] = np.argmax(np.random.multinomial(1, self.p0, size=1))
        # Uniform transition matrix
        A1 = np.zeros((self.Q, self.Q))
        A1[(np.arange(0, self.Q - 1), np.arange(0, self.Q - 1))] = 1 - self.Q / self.T
        A1[(np.arange(0, self.Q - 1), np.arange(1, self.Q))] = self.Q / self.T
        A1[-1, -1] = 1

        for ii in range(self.T - 1):
            trace[ii + 1] = np.argmax(np.random.multinomial(1, A1[int(trace[ii]), :], size=1))
        return trace

    def eval(self, x):
        self.T = x.shape[0]
        self.set_likelihood_mtx(x)

        trace, log_like = self.viterbi(x)
        return trace, log_like

    def viterbi(self, x):

        # Viterbi initialization
        self.delta = np.zeros((self.Q, self.T))
        self.phi = np.zeros((self.Q, self.T))
        # clausula magica

        non_0_p0_idx = (self.p0 != 0)
        self.phi[:, 0] = -1 * np.inf
        # minus B because B is -loglike
        self.phi[non_0_p0_idx, 0] = np.log(self.p0[non_0_p0_idx]) + self.B[non_0_p0_idx, 0]

        for ti in np.arange(1, self.T):
            self.prune(ti)

        trace = self.backtracking(self.phi, self.delta)
        # flip becuase backtracking starts at end
        trace = np.flipud(trace)

        return trace, np.amax(self.phi[:, self.T - 1])

    def prune(self, t):
        # qi current state
        for qi in range(self.Q):
            phi_t = np.zeros(self.Q)
            # qj previous state
            for qj in range(self.Q):
                phi_t[qj] = self.phi[qj, t - 1] + (np.log(self.A[qj, qi]) if self.A[qj, qi] != 0 else np.inf * -1)
                # print('self.phi[qj,t-1]',self.phi[qj,t-1])

            # print('phi_t',phi_t)
            self.delta[qi, t] = np.argmax(phi_t)
            self.phi[qi, t] = np.amax(phi_t) + self.B[qi, t]

    def save(self, where):
        file = open(where, 'wb')
        file.write(cPickle.dumps(self.__dict__))
        file.close()

    def load(self, where):
        file = open(where, 'rb')
        dataPickle = file.read()
        file.close()

        self.__dict__ = cPickle.loads(dataPickle)