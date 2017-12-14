#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
cimport numpy as np
import numpy as np

cpdef double optimize(
        np.ndarray[np.float_t, ndim=2] K,
        np.ndarray[np.float_t, ndim=1] y,
        np.ndarray[np.float_t, ndim=1] coef,
        double C,
        double tolerance,
        int passes_limit,
        int iterate_limit):

    cdef int n_samples = K.shape[0]
    cdef int it = 0, passes = 0
    cdef int alphas_changed
    cdef int i, j
    cdef double Ei, Ej, ai, aj, newai, newaj, eta, L, H, b1, b2, yEi
    cdef object random_state = np.random.RandomState(0)

    cdef np.ndarray[np.float_t, ndim=2] yK = y * K

    cdef double b = 0.0
    while passes < passes_limit and it < iterate_limit:
        alphas_changed = 0
        for i in range(n_samples):
            Ei = margins_kernel(yK[i], coef, b) - y[i]
            yEi = y[i] * Ei
            if (yEi < -tolerance and coef[i] < C) or (yEi > tolerance and coef[i] > 0):
                # alphas[i] needs updating! Pick a j to update it with
                j = i
                while j == i:
                    j = random_state.randint(n_samples)
                Ej = margins_kernel(yK[j], coef, b) - y[j]

                # compute L and H bounds for j to ensure we're in [0, C]x[0, C]
                ai = coef[i]
                aj = coef[j]
                if y[i] == y[j]:
                    L = max(0, ai + aj - C)
                    H = min(C, ai + aj)
                else:
                    L = max(0, aj - ai)
                    H = min(C, aj - ai + C)

                if abs(L - H) < 1e-4:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                # compute new alpha[j] and clip it inside [0 C]x[0 C]
                # box then compute alpha[i] based on it.
                newaj = aj - y[j] * (Ei - Ej) / eta
                newaj = min(newaj, H)
                newaj = max(newaj, L)
                if abs(aj - newaj) < 1e-4:
                    continue
                coef[j] = newaj
                newai = ai + y[i] * y[j] * (aj - newaj)
                coef[i] = newai

                # update the bias term
                b1 = (b - Ei - y[i] * (newai - ai) * K[i, i] -
                      y[j] * (newaj - aj) * K[i, j])
                b2 = (b - Ej - y[i] * (newai - ai) * K[i, j] -
                      y[j] * (newaj - aj) * K[j, j])
                b = 0.5 * (b1 + b2)

                if 0 < newai < C:
                    b = b1
                elif 0 < newaj < C:
                    b = b2

                alphas_changed += 1

        it += 1

        if alphas_changed == 0:
            passes += 1
        else:
            passes = 0

    return b


cdef double margins_kernel(
        np.ndarray[np.float_t, ndim=1] yk,
        np.ndarray[np.float_t, ndim=1] dual_coef,
        double intercept):
    return intercept + np.dot(dual_coef, yk)