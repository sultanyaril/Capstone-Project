from main import *
from multiprocessing import Pool, Array
from sklearn.metrics import confusion_matrix
from welford import Welford

H = 10 ** (-7) # X time
h = 10 ** (-7) # Y time
window = 10 ** (-4)
h_f = 10**(-4)
T = 1# in minutes

seed = 0
if seed != 0:
    np.random.seed(seed)


dim_X = 4
dim_Y = 1

Lambda = np.array([[-12.5, 12.5, 0, 0],
                    [0, -1000, 1000, 0],
                    [0, 0, -250, 250],
                    [40, 0, 10, -50]])
pi = stationary_distr(np.eye(dim_X) + H * Lambda)
f = np.array([[0.07],
              [0.03],
              [0.02],
              [0.025]])
g = np.array([np.diag([0.1]),
              np.diag([0.5]),
              np.diag([0.6]),
              np.diag([0.3])])
weibull_distr_params = np.array([[1, 0.0000060],  # in article we have alpha = 1/Lambda, beta = k
                                 [1.2, 0.0000050],  # k - wiki, scale
                                 [1.2, 0.0000055],  #
                                 [1.4, 0.0000070]])
markov_chain = (Lambda, pi, f, g)
discretization = (H, h)
dimensions = (dim_X, dim_Y)


N = 100


def one_iteration(i):
    print(i)

    X, X_grid = model_X(dim_X, markov_chain, H, T)
    Y, Y_sum, Y_grid = model_Y(X, dim_Y, markov_chain, discretization, T)

    S, t = model_observations(Y_sum, X, weibull_distr_params, discretization, Y_grid, dimensions)
    mean_S, mean_t, cumm_T = model_trades(S, t, window, T)
    S = 0
    t = 0
    U = np.vstack([mean_t, mean_S]).T

    X_random_time = filter_random_time(U, h, window, markov_chain, T, weibull_distr_params, dim_X)
    X_discrete_time, est_grid = filter_discrete_time(Y_sum, dimensions, markov_chain, (H, window), T)
    X_vectorized = vectorize(X, dim_X)
    return Welford(X_discrete_time), \
           Welford(X_random_time), \
           Welford(X_vectorized - X_discrete_time), \
           Welford(X_vectorized - X_random_time)

if __name__ == '__main__':
    w_discrete_time, w_random_time, w_error_discrete, w_error_random = one_iteration(0)
    acc_random_time = 0
    acc_discrete_time = 0

    with Pool(6) as pool:
        jobs = []
        for i in range(N-1):
            result = pool.apply_async(one_iteration, args = (i, ))
            jobs.append(result)

        for i in range(N-1):
            tmp_discrete, tmp_random, tmp_error_discrete, tmp_error_random = jobs[i].get()
            w_discrete_time.merge(tmp_discrete)
            w_random_time.merge(tmp_random)
            w_error_discrete.merge(tmp_error_discrete)
            w_error_random.merge(tmp_error_random)
    print(np.sqrt(w_discrete_time.var_s))
    print(np.sqrt(w_random_time.var_s))
    print(np.sqrt(w_error_discrete.var_s))
    print(np.sqrt(w_error_random.var_s))
    np.savez('stds.npz', discrete_time=np.sqrt(w_discrete_time.var_s), \
                         random_time=np.sqrt(w_random_time.var_s), \
                         error_discrete=np.sqrt(w_error_discrete.var_s), \
                         error_random=np.sqrt(w_error_random.var_s))
    print('FINISH SUCCESS')



