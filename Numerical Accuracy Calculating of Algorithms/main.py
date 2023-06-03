import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import multivariate_normal, weibull_min
from tqdm import tqdm

def stationary_distr(Q):
  evals, evecs = np.linalg.eig(Q.T)
  evec1 = evecs[:,np.isclose(evals, 1)]

#Since np.isclose will return an array, we've indexed with an array
#so we still have our 2nd axis.  Get rid of it, since it's only size 1.
  evec1 = evec1[:,0]

  stationary = evec1 / evec1.sum()

#eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
  stationary = stationary.real
  return stationary


def init_X(pi, U=np.random.uniform(0, 1, 1)):
  cumm = 0
  for i in range(len(pi)):
    cumm += pi[i]
    if cumm > U:
      return i


def scale(row):
  s = np.sum(row)
  return row/s


def model_X(dim_X, markov_chain, H, T):
  Lambda, pi, f, g = markov_chain
  
  # state
  X_grid_length = int(T/H)
  U = np.random.uniform(0, 1, X_grid_length)
  X_grid = np.arange(0, X_grid_length)
  X = np.empty(X_grid_length, dtype = np.int8)
  X[0] = init_X(pi)
  P = np.eye(dim_X) + H * Lambda
  for i in tqdm(X_grid[1:], leave=False):
    P_i = P[X[i - 1]]
    X[i] = init_X(scale(P_i), U[i])
  return X, X_grid


def model_Y(X, dim_Y, markov_chain, discretization, T):
  H, h = discretization
  Lambda, pi, f, g = markov_chain
  # observations
  Y_grid_length = int(T/h)

  W =  np.random.multivariate_normal(np.zeros(dim_Y), np.eye(dim_Y), Y_grid_length)

  Y_grid = np.arange(0, Y_grid_length)
  Y = np.empty((Y_grid_length, dim_Y))
  S = np.empty((Y_grid_length, dim_Y))
  S[0] = np.ones(dim_Y)
  for i in tqdm(Y_grid[1:], leave=False):
    Y[i] = np.diag(S[i-1]) @ f[X[i * int(h/H)]] * h + np.diag(S[i-1]) @ W[i] @ np.linalg.cholesky(h * g[X[i * int(h/H)]])
    S[i] = S[i - 1] + Y[i]
  return Y, S, Y_grid  


def to_sums(Y, start=0):
  sum_Y = np.empty(Y.size)
  sum_Y[0] = start
  for i in tqdm(range(sum_Y.size)[1:], leave=False):
    sum_Y[i] = sum_Y[i - 1] + Y[i - 1]
  return sum_Y


def model_observations(Y, X, weibull_distr_params, discretization, Y_grid, dimensions):
  dim_X, dim_Y = dimensions
  H, h = discretization
  i = 0
  t = np.empty(Y_grid.size)
  S = np.empty(Y_grid.size)
  curr_pos = 0
  pbar = tqdm(total=Y_grid.size, leave=False)
  while i < Y_grid.size:
    prev = i
    weibull = weibull_min.rvs(weibull_distr_params[X[i * int(h/H)]][0], size=1, scale=weibull_distr_params[X[i * int(h/H)]][1])
    w_int = int(np.around(weibull * h ** (-1), 0))
    weibull_round = w_int * h
    t[curr_pos] = weibull_round
    i += w_int
    if i < len(Y_grid):
      S[curr_pos] = np.log(Y[i] / Y[prev])
    curr_pos += 1
    pbar.update(i - prev)
  pbar.close()
  return S[:curr_pos - 1], t[:curr_pos - 1]


def model_trades(S, t, window_orig, T):
  mean_S = np.empty(int(T/window_orig))
  mean_t = np.empty(int(T/window_orig))
  cumm_T = np.empty(int(T/window_orig))
  index_of_left = 0
  curr_index = 0
  window = window_orig
  for i in tqdm(range(t.size), leave=False):
    if np.sum(t[index_of_left:i]) >= window:
      mean_S[curr_index] = np.sum(S[index_of_left:i]) / np.sqrt(window)  # or should be window?
      mean_t[curr_index] = (i - index_of_left) / np.sqrt(window)  # np.sum(t[index_of_left:i]) / np.sqrt(window_orig) 
      cumm_T[curr_index] = i - index_of_left
      window = window + window_orig - np.sum(t[index_of_left:i])
      index_of_left = i
      curr_index += 1
  left_time = np.sum(t[index_of_left:])
  mean_S[curr_index] = np.sum(S[index_of_left:]) / np.sqrt(left_time)
  mean_t[curr_index] = (t.size - index_of_left) / np.sqrt(left_time)
  cumm_T[curr_index] = t.size - index_of_left
  return mean_S, mean_t, cumm_T



def filter_random_time(U, h, window, markov_chain, T, weibull_distr_params, dim_X):
  Lambda, pi, f, g = markov_chain
  g = g.flatten()
  f = f.flatten()
  X_pred_grid_length = int(T/h)
  X_pred = np.empty((X_pred_grid_length, dim_X), dtype=np.float32)
  X_pred_grid = np.arange(0, X_pred_grid_length)
  N_0 = np.empty((dim_X, 2))
  N_1 = np.empty((dim_X, 2, 2))
  for i in range(dim_X):
    m_l = weibull_min.mean(weibull_distr_params[i][0], scale=weibull_distr_params[i][1])
    d_l = weibull_min.var(weibull_distr_params[i][0], scale=weibull_distr_params[i][1])
    N_0[i] = np.array([np.sqrt(window) / m_l, (f[i] - g[i] / 2) * np.sqrt(window)])
    N_1[i] = np.diag([d_l / (m_l ** 3), g[i]])  # m_l or a_l?
  
  P_T = (np.eye(dim_X) + h * Lambda).T
  X_pred[0] = pi
  for i in tqdm(X_pred_grid[1:], leave=False):
    X_pred[i] = P_T @ X_pred[i-1]
    if i % (int(window / h)) == 0:
      Z = np.empty((dim_X))
      for j in range(dim_X):
        Z[j] = X_pred[i][j] * multivariate_normal.pdf(U[i // int((window / h))], N_0[j], N_1[j])
      if np.sum(Z) == 0:
        print(i)
      X_pred[i] = Z / np.sum(Z)
  return X_pred


def filter_discrete_time(Y, dimensions, markov_chain, discretization, T):
  # filtration
  dim_X, dim_Y = dimensions
  Lambda, pi, f, g = markov_chain
  H, h = discretization
  est_grid_length = int(T/H)
  estimation = np.empty((est_grid_length, dim_X), dtype = np.dtype('f8'))
  est_grid = np.arange(0, est_grid_length)
  
  N = multivariate_normal.pdf

  estimation[0] = pi#estimation_0
  for i in tqdm(est_grid[1:], leave=False):
    forecast = (np.eye(dim_X) + H * Lambda.T) @ estimation[i - 1]
    estimation[i] = forecast
    if i % (int(h/H)) == 0:
        k_t = np.diag([N(Y[i] - Y[i - int(h/H)], h * f[j], h * g[j]) for j in range(dim_X)])
        ones = np.ones((1, dim_X))
        estimation[i] = 1 / (ones @ k_t @ forecast) * k_t @ forecast
  return estimation, est_grid

def vectorize(X, dim_X):
    X_vector = np.empty((X.size, dim_X))
    for i in tqdm(np.arange(X.size), leave=False):
        X_vector[i] = np.array([1 if j == X[i] else 0 for j in range(dim_X)])
    return X_vector