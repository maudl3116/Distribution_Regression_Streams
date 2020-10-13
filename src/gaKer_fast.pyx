# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as np
from libc.float cimport DBL_MAX
from libc.math cimport exp, log
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef inline double _softmin3(DTYPE_t a,
                             DTYPE_t b,
                             DTYPE_t c,
                             DTYPE_t gamma):
    a /= -gamma
    b /= -gamma
    c /= -gamma

    cdef DTYPE_t max_val = max(max(a, b), c)

    cdef DTYPE_t tmp = 0
    tmp += exp(a - max_val)
    tmp += exp(b - max_val)
    tmp += exp(c - max_val)

    return -gamma * (log(tmp) + max_val)


def GA_kernels_mmd(double[:,:,:] x, double[:,:,:] y, double gamma):
    cdef int A = x.shape[0]
    cdef int B = y.shape[0]
    cdef int M = x.shape[1]
    cdef int N = y.shape[1]
    cdef int D = x.shape[2]
    cdef double eucli_dist
    cdef int i, j, k, l
    cdef double[:,:,:,:] R = np.zeros((A,B,M+2,N+2), dtype=np.float64)

    for l in range(A):
        for m in range(B):

            for i in range(M+1):
                R[l,m,i, 0] = DBL_MAX

            for j in range(N+1):
                R[l,m,0, j] = DBL_MAX

            R[l,m,0, 0] = 0

            for i in range(1,M+1):
                for j in range(1,N+1):
                    eucli_dist = 0.
                    for k in range(D):
                        eucli_dist += (x[l,i-1,k]-y[m,j-1,k])*(x[l,i-1,k]-y[m,j-1,k])
                    R[l,m,i, j] = eucli_dist + _softmin3(R[l,m,i-1, j],R[l,m,i-1, j-1],R[l,m,i, j-1],gamma)

    return R[:,:,M,N]