"""
Thomas Kahn
thomas.b.kahn@gmail.com
"""
import numpy as np
from math import sqrt
import multiprocessing as mp


def t_scan(L, window = 1e3):
    """
    Computes t statistic for i to i+window points versus i-window to i
    points for each point i in input array. Uses multiple processes to
    do this calculation asynchronously. Array is decomposed into window
    number of frames, each consisting of points spaced at window
    intervals. This optimizes the calculation, as the drone function
    need only compute the mean and variance for each set once.

    Parameters
    ----------
    L : numpy array
        1 dimensional array that represents time series of datapoints
    window : int / float
        Number of points that comprise the windows of data that are
        compared


    Returns
    -------
    t_stat : numpy array
        Array which holds t statistic values for each point. The first 
        and last (window) points are replaced with zero, since the t
        statistic calculation cannot be performed in that case.

    """
    size     = L.size
    window   = int(window)
    frames   = range(window)
    n_cols   = (size / window) - 1
    
    t_stat  = np.zeros((window, n_cols))
    pool    = mp.Pool(processes = mp.cpu_count() - 1)
    results = [pool.apply_async(t_scan_drone, args=(L, n_cols, frame, window)) for frame in frames]
    results = [r.get() for r in results]
    
    for index, row in results:
        t_stat[index] = row
    
    t_stat = np.append(np.zeros(window), np.append(t_stat.transpose(), np.zeros(size % window))) # calling np.append with no axis causes raveling

    return t_stat


def t_scan_drone(L, n_cols, frame, window=1e3):
    size   = L.size
    window = int(window)
    root_n = sqrt(window)

    output = np.zeros(n_cols)
    b      = L[frame:window+frame]
    b_mean = b.mean()
    b_var  = b.var()
    for i in range(window+frame, size-window, window):
        a = L[i:i+window]
        a_mean = a.mean()
        a_var  = a.var()
        output[i / window - 1] = root_n * (a_mean - b_mean) / sqrt(a_var + b_var)
        b_mean, b_var = a_mean, a_var

    return frame, output


def mz_fwt(x, n=2):
    N_pnts = x.size
    lambda_j = [1.5, 1.12, 1.03, 1.01][0:n]
    if n > 4:
        lambda_j += [1.0]*(n-4)
    
    H = np.array([0.125, 0.375, 0.375, 0.125])
    G = np.array([2.0, -2.0])
    
    Gn = [2]
    Hn = [3]
    for j in range(1,n):
        q = 2**(j-1)
        Gn.append(q+1)
        Hn.append(3*q+1)

    S = np.append(x[::-1], x)
    S = np.append(S, x[::-1])
    prod = np.ones(N_pnts)
    for j in range(n):
        n_zeros = 2**j - 1
        Gz = insert_zeros(G, n_zeros)
        Hz = insert_zeros(H, n_zeros)
        current = (1.0/lambda_j[j])*np.convolve(S,Gz)
        current = current[N_pnts+Gn[j]:2*N_pnts+Gn[j]]
        prod *= current
        S_new = np.convolve(S, Hz)
        S_new = S_new[N_pnts+Hn[j]:2*N_pnts+Hn[j]]
        S = np.append(S_new[::-1], S_new)
        S = np.append(S, S_new[::-1])
    return prod



def insert_zeros(x, n):
    newlen  = (n+1)*x.size
    out     = np.zeros(newlen)
    indices = range(0, newlen-n, n+1)
    out[indices] = x
    return out



def find_steps(array, threshold):
    steps = []
    above_points = np.where(array > threshold, 1, 0)
    ap_dif = np.diff(above_points)
    cross_ups = np.where(ap_dif == 1)[0]
    cross_dns = np.where(ap_dif == -1)[0]
    for upi, dni in zip(cross_ups,cross_dns):
        steps.append(np.argmax(array[upi:dni]) + upi)
    return steps



def get_step_sizes(array, indices, window=1000):
    step_sizes = []
    step_error = []
    indices = sorted(indices)
    last = len(indices) - 1
    for i, index in enumerate(indices):
        if i == 0:
            q = min(window, indices[i+1]-index)
        elif i == last:
            q = min(window, index - indices[i-1])
        else:
            q = min(window, index-indices[i-1], indices[i+1]-index)
        a = array[index:index+q]
        b = array[index-q:index]
        step_sizes.append(a.mean() - b.mean())
        step_error.append(sqrt(a.var()+b.var()))
    return step_sizes, step_error