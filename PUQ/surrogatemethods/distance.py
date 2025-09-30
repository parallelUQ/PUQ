import numpy as np

# def distcpp(X1):
#     nr = X1.shape[0]
#     nc = X1.shape[1]
#     s = np.zeros((nr, nr))
#     for i in range(1, nr):
#         ptrs = s[:, i]
#         ptrs2 = s[i, :]  # symmetric
#         for j in range(i):
#             ptrX1 = X1[i, :]
#             ptrX2 = X1[j, :]
#             tmp = np.abs(ptrX1 - ptrX2)
#             ptrs += tmp * tmp
#             ptrs2[:] = ptrs
#             ptrs2 += nr

#     print(np.round(s))
#     return s

# def distcpp(X1):
#     nr, nc = X1.shape
#     s = np.zeros((nr, nr))

#     for i in range(1, nr):
#         ptrs = s[0:i, i]
#         ptrs2 = s[i, 0:i]

#         for j in range(i):
#             ptrX1 = X1[i, :]
#             ptrX2 = X1[j, :]
#             tmp = np.sum((ptrX1 - ptrX2) ** 2)
#             ptrs += tmp
#             ptrs2[j] = tmp

#     return s

import numpy as np

# def distcpp(X1):
#     nr, nc = X1.shape
#     s = np.zeros((nr, nr))

#     for i in range(1, nr):
#         ptrs = s[0:i, i]
#         ptrs2 = s[i, 0:i]

#         for j in range(i):
#             ptrX1 = X1[i, :]
#             ptrX2 = X1[j, :]
#             tmp = np.sum((ptrX1 - ptrX2) ** 2)
#             ptrs[j] = tmp
#             ptrs2[j] = tmp

#     # Make the matrix symmetric
#     s = s + s.T - np.diag(s.diagonal())

#     return s

def distcpp(X1, X2=None):
    if X2 is None:
        return np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X1**2, axis=1) - 2 * np.dot(X1, X1.T)
    else:
        return np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X1**2, axis=1) - 2 * np.dot(X1, X1.T)
        
def distcpp_2(X1, X2):
    nr1 = X1.shape[0]
    nr2 = X2.shape[0]
    dim = X1.shape[1]
    s = np.zeros((nr1, nr2))
    ptrs = s.flatten()
    ptrX2 = X2.flatten()
    ptrX1 = X1.flatten()
    for i in range(nr2):
        for j in range(nr1):
            tmp = np.abs(ptrX1 - ptrX2)
            ptrs += tmp * tmp
            ptrX2 -= nr2 * dim
            ptrX1 -= nr1 * dim - 1

        ptrX2 += 1
        ptrX1 -= nr1

    return s

def distcppMaha(X1, m):
    nr = X1.shape[0]
    nc = X1.shape[1]
    s = np.zeros((nr, nr))
    for i in range(1, nr):
        ptrs = s[:, i]
        ptrs2 = s[i, :]  # symmetric
        ptrX1 = X1[i, :]
        ptrm = m
        for j in range(i):
            ptrX2 = X1[j, :]
            tmp = np.abs(ptrX1 - ptrX2)
            ptrs += tmp * tmp / ptrm
            ptrX1 += nr
            ptrX2 += nr
            ptrm += 1

        ptrs2[:] = ptrs
        ptrs2 += nr

    return s

def distcppMaha_2(X1, X2, m):
    nr1 = X1.shape[0]
    nr2 = X2.shape[0]
    dim = X1.shape[1]
    s = np.zeros((nr1, nr2))
    ptrs = s.flatten()
    ptrX2 = X2.flatten()
    ptrX1 = X1.flatten()
    ptrm = m
    for i in range(nr2):
        for j in range(nr1):
            tmp = np.abs(ptrX1 - ptrX2)
            ptrs += tmp * tmp / ptrm
            ptrX2 -= nr2 * dim
            ptrX1 -= nr1 * dim - 1
            ptrm -= dim

        ptrX2 += 1
        ptrX1 -= nr1
        ptrm += dim

    return s

def distance_cpp(X1, X2=None, m=None):
    if X2 is not None:
        if m is not None:
            return distcppMaha_2(X1, X2, m)
        else:
            return distcpp_2(X1, X2)
    else:
        if m is not None:
            return distcppMaha(X1, m)
        else:
            return distcpp(X1)
