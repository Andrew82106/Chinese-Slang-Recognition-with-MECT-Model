import numpy as np
from scipy.spatial.distance import cdist


def make_dist_matrix(Q, C):
    M = np.zeros((len(Q) + 1, len(C) + 1))
    M[0, 1:] = float("inf")
    M[1:, 0] = float("inf")
    M[1:, 1:] = cdist([[q] for q in Q], [[c] for c in C], 'euclidean')
    return M


def naive_DTW(Q, C, fun=make_dist_matrix):
    M1 = fun(Q, C)
    M2 = M1[1:, 1:]  # 浅copy
    for i in range(len(Q)):
        for j in range(len(C)):
            M2[i, j] += min(M1[i, j], M1[i + 1, j], M1[i, j + 1])  # 刷DP表
    return M2[-1, -1]


if __name__ == "__main__":
    lst1 = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 2, 2, 1, 1, 1]
    lst2 = [4, 4, 4, 5, 5, 5, 6, 6, 6, 5, 5, 4, 4]
    lst3 = [0, 0, 0, 0, 1, 1, 1, 2, 2, 1, 1, 0, 0]
    lst4 = [2, 2, 2, 3, 3, 2, 2, 1, 1, 3, 3, 3, 3]

    lst = [lst1, lst2, lst3, lst4]

    for i in range(0, 4):
        for j in range(i, 4):
            print(f"{i}-{j}:{naive_DTW(lst[i], lst[j])}")
