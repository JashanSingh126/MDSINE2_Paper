#computes the spearman corrreation for PNA

import numpy as np
from scipy import stats

def remove_nan(v1, v2, N):
    """
    get rid of nans in the arrays v1 and v2 of length N

    """

    new_v1 = []
    new_v2 = []
    for i in range(N):
        if not np.isnan(v1[i]) and not np.isnan(v2[i]):
            new_v1.append(v1[i])
            new_v2.append(v2[i])

    return np.asarray(new_v1), np.asarray(new_v2)

def compute_average_spearman(A, B, statistic):
    """
    returns the Spearman Rank coefficient between matrices A and B
    """

    if A.shape != B.shape:
        print("Error. The shapes of the matrices must be equal")

    all_sp = []
    n_row = A.shape[0]
    n_col = A.shape[1]
    for i in range(n_row):
        vec_a, vec_b = remove_nan(A[i], B[i], n_col)
        #print(vec_a, vec_b)
        sp = stats.spearmanr(vec_a, vec_b)[0]
        #print(sp)
        if not np.isnan(sp):
            all_sp.append(sp)

    if statistic == "mean":
        return np.mean(all_sp)
    else:
        return np.median(all_sp)
