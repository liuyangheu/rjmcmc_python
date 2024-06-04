# -*- encoding: utf-8 -*-
'''
@File    :   curvefit.py
@Author  :   Yang Liu (yang.liu3@halliburton.com) 
'''
# import lib below

from math import sqrt, exp, pi
from rjmcmc_util import *

#check all excpet last few lines
class CurveFitResult:
    def __init__(self):
        self.maxorder = 0
        self.alpha = []
        self.beta = []
        self.L = []
        self.Z = []
        self.S = []
        self.Si = []
        self.mu = []
        self.x = []
        self.b = []

def curvefit_create(maxorder):
    cf = CurveFitResult()
    cf.maxorder = maxorder
    cf.alpha = [0.0] * (2 * (maxorder + 1))
    cf.beta = [0.0] * (2 * (maxorder + 1))
    cf.L = [[0.0] * (maxorder + 1) for _ in range(maxorder + 1)]
    cf.Z = [[0.0] * (maxorder + 1) for _ in range(maxorder + 1)]
    cf.S = [[0.0] * (maxorder + 1) for _ in range(maxorder + 1)]
    cf.Si = [[0.0] * (maxorder + 1) for _ in range(maxorder + 1)]
    cf.mu = [0.0] * (maxorder + 1)
    cf.x = [0.0] * (2 * (maxorder + 1))
    cf.b = [0.0] * (2 * (maxorder + 1))
    return cf

def curvefit_compute(d, di, dj, order, cf):
    n = dj - di + 1
    if n < order:
        return -1
    # Update cf.alpha and cf.beta
    compute_hankel(d.points[di:], n, 1.0, order, cf)
    # Update cf.L
    if compute_hankel_cholesky(cf.alpha, order, cf.L) < 0:
        return -1
    # Update cf.Si
    compute_square(cf.L, order, cf.Si)
    # Update of cf.x
    compute_forward_substitution(cf.L, order, cf.beta, cf.x)
    # Update of cf.mu
    compute_backward_substitution(cf.L, order, cf.x, cf.mu)
    # Update cf.Z, cf.b, cf.x
    compute_inverse(cf.L, order, cf.Z, cf.b, cf.x)
    # Update cf.S
    compute_square(cf.Z, order, cf.S)
    # Update cf.Z
    if compute_cholesky(cf.S, order, cf.Z) < 0:
        return -1
    return 0

def curvefit_compute_lambda(d, lambda_, di, dj, order, cf):
    if cf is None:
        rjmcmc_error("curvefit_compute_lambda: result not allocated")
        return -1
    if order > cf.maxorder:
        rjmcmc_error(f"curvefit_compute_lambda: requested order is too large ({order} > {cf.maxorder})")
        return -1
    if dj <= di:
        rjmcmc_error(f"curvefit_compute_lambda: invalid range ({di} {dj})")
        return -1
    # regression coefficients are stored in cf.beta
    n = dj - di + 1
    if n < order:
        rjmcmc_error("curvefit_compute_lambda: insufficient points")
        return -1
    # Update Hankel matrix coefficients cf.alpha and cf.beta
    compute_hankel(d.points[di:], n, lambda_, order, cf)
    # Update cf.L, lower triangular matrix
    # Cholesky decomposition 
    if compute_hankel_cholesky(cf.alpha, order, cf.L) < 0:
        rjmcmc_error("curvefit_compute_lambda: failed to compute hankel cholesky")
        return -1
    # Update of cf.x
    compute_forward_substitution(cf.L, order, cf.beta, cf.x)
    # Update of cf.mu
    compute_backward_substitution(cf.L, order, cf.x, cf.mu)
    # Update cf.Z, cf.b, cf.x
    compute_inverse(cf.L, order, cf.Z, cf.b, cf.x)
    # Update cf.S
    compute_square(cf.Z, order, cf.S)
    # Update cf.Z
    if compute_cholesky(cf.S, order, cf.Z) < 0:
        rjmcmc_error("curvefit_compute_lambda: failed to compute cholesky")
        return -1
    return 0

def curvefit_sample(cf, normal, coeff, coeff_len):
    m = coeff_len
    # generate random noise added to the mean value of fitting coefficient
    for i in range(m):
        cf.x[i] = normal()
    # update regression coefficient
    for i in range(m):
        coeff[i] = cf.mu[i]
        for j in range(m):
            coeff[i] += cf.x[j] * cf.Z[i][j]
        cf.b[i] = coeff[i] - cf.mu[i]
    # likelihood of a multivariate normal distribution using the updated coefficient
    sigma2 = 0.0 # variance of residual (difference between curve and data)
    for i in range(m):
        sigma2t = 0.0
        for j in range(m):
            sigma2t += cf.Si[i][j] * cf.b[j]
        sigma2 += cf.b[i] * sigma2t
    sigmah = 1.0 # std of the random coefficients
    for i in range(m):
        sigmah *= cf.Z[i][i]
    if sigmah < 0.0:
        return -1
    # likelihood of a multivariate normal distribution
    prob = exp(-0.5 * sigma2) / ((2.0 * pi)**(m / 2.0) * (1.0 / sigmah))
    # prob = exp(-0.5 * sigma2) / (pow(2.0 * M_PI, m / 2.0) * (1.0 / sigmah))
    return 0, prob

def curvefit_sample_mean(cf, coeff, coeff_len):
    for i in range(coeff_len):
        coeff[i] = cf.mu[i]
    return 0

def curvefit_sample_sigma(cf, sigma, sigma_len):
    for i in range(sigma_len):
        sigma[i] = sqrt(cf.S[i][i])
    return 0

def curvefit_sample_detCm(cf, detCm, order):
    sigmah = 1.0
    for i in range(order + 1):
        sigmah *= cf.Z[i][i]
    # detCm[order] = sigmah * sigmah
    detCm[0] = sigmah * sigmah
    return 0

def compute_hankel(points, n, lambda_, order, cf):
    m = 2 * (order + 1)
    cf.alpha = [0.0] * m
    cf.beta = [0.0] * m
    l2 = lambda_ * lambda_
    for i in range(n):
        ai = 1.0 / (l2 * points[i].n * points[i].n)*points[i].w
        bi = ai * points[i].y
        cf.alpha[0] += ai
        cf.beta[0] += bi
        for j in range(1, m):
            ai *= points[i].x
            bi *= points[i].x
            cf.alpha[j] += ai
            cf.beta[j] += bi
    return 0

def set_I(m, n):
    for i in range(n):
        for j in range(n):
            m[i][j] = 1.0 if i == j else 0.0

def compute_hankel_cholesky(alpha, order, L):
    m = order + 1
    set_I(L, m)
    for j in range(m):
        L[j][j] = alpha[2 * j]
        for k in range(j):
            L[j][j] -= L[j][k] * L[j][k]
        # comment out to avoid numerical stability
        # if L[j][j] <= 0.0:
        #     return -1
        L[j][j] = sqrt(abs(L[j][j]))
        for i in range(j + 1, m):
            L[i][j] = alpha[i + j]
            for k in range(j):
                L[i][j] -= L[i][k] * L[j][k]
            L[i][j] /= L[j][j]
    return 0

def compute_forward_substitution(L, order, b, x):
    m = order + 1
    for j in range(m):
        x[j] = b[j]
        for i in range(j):
            x[j] -= L[j][i] * x[i]
        x[j] /= L[j][j]

def compute_backward_substitution(L, order, b, x):
    m = order + 1
    for j in range(m - 1, -1, -1):
        x[j] = b[j]
        for i in range(m - 1, j, -1):
            x[j] -= L[i][j] * x[i]
        x[j] /= L[j][j]

def compute_inverse(L, order, Z, b, x):
    m = order + 1
    for j in range(m):
        for i in range(m):
            b[i] = 1.0 if j == i else 0.0
        compute_forward_substitution(L, order, b, x)
        for i in range(m):
            Z[j][i] = x[i]
    return 0

def compute_square(L, order, L2):
    m = order + 1
    for i in range(m):
        for j in range(m):
            t = 0.0
            for k in range(m):
                t += L[i][k] * L[j][k]
            L2[i][j] = t
    return 0

def compute_cholesky(A, order, L):
    m = order + 1
    set_I(L, m)
    for j in range(m):
        L[j][j] = A[j][j]
        for k in range(j):
            L[j][j] -= L[j][k] * L[j][k]
        # if L[j][j] <= 0.0:
        #     return -1
        L[j][j] = sqrt(abs(L[j][j]))
        for i in range(j + 1, m):
            L[i][j] = A[i][j]
            for k in range(j):
                L[i][j] -= L[i][k] * L[j][k]
            L[i][j] /= L[j][j]
    return 0

def fx(a, order, x):
    xp = 1.0
    y = 0.0
    for i in range(order + 1):
        y += xp * a[i]
        xp *= x
    return y

# Curve-fitting algorithm: misfit of the curve to the data
def curvefit_compute_mean_misfit(cf, data, di, dj, lambda_, order, mean, sigma, mean_misfit, detCm):
    # di, dj: range of the data points (partition)
    # mean, sigma: mean and standard deviation of the curve fit
    # detCm: determinant of the covariance matrix
    if curvefit_compute_lambda(data, lambda_, di, dj, order, cf) < 0:
        return -1
    if curvefit_sample_mean(cf, mean, order + 1) < 0:
        return -1
    if curvefit_sample_sigma(cf, sigma, order + 1) < 0:
        return -1
    if curvefit_sample_detCm(cf, detCm, order)< 0:
        return -1
    misfit = 0.0
    
    for i in range(di, dj + 1):
        dy = fx(mean, order, data.points[i].x) - data.points[i].y
        n = data.points[i].n
        # misfit += (dy * dy) / (2.0 * n * n)
        misfit += (data.points[i].w*dy * dy) / (2.0 * n * n )
    mean_misfit[0] = misfit
    return 0

def curvefit_evaluate_pk(cf, data, di, dj, max_order, fixed_prior, auto_z, mean_misfit, detCm, autoprior, S, pk, kcdf, mean, sigma):
    # iterate over orders from 0 to 'max_order'
    # 1. curve fit properties of each order
    for k in range(max_order + 1):
        # average value of the discrepancy between model(curve) and data: (1,max_order)
        mean_misfit[k] = 0.0
        # determinants of covariance matrices: (1, max_order) 
        detCm[k] = 0.0
        # update of the curve fit properties: mean, sigma, detCm, mean_misfit
        if curvefit_compute_mean_misfit(cf, data, di, dj, 1.0, k, mean[k], sigma[k], mean_misfit[k:], detCm[k:]) < 0:
            print('mean_misfit')
            return -1
    
    # 2.order probability
    # specify prior belief or preference
    if fixed_prior is not None:
        for k in range(max_order + 1):
            if k == 0:
                autoprior[k] = fixed_prior[k]
            else:
                autoprior[k] = autoprior[k - 1] * fixed_prior[k]
    else:
    # derive priors based on the statistical characteristics of the data
        for k in range(max_order + 1):
            autoprior[k] = 1.0
            for j in range(k + 1):
                # sigma: matrix of variances and covariances
                autoprior[k] *= (sigma[k][j] * 2.0 * auto_z) # auto_z:scaling factor 
    for k in range(max_order + 1):
        # square matrix to store conditional probabilities of one order given another (different from cf.S)
        S[k][k] = 1.0
        if detCm[k] > 0.0:
            for j in range(k + 1, max_order + 1):
                if detCm[j] > 0.0:
                    S[k][j] = exp(mean_misfit[j] - mean_misfit[k]) * sqrt(pow(2.0 * pi, k - j) * detCm[k] / detCm[j])
                    S[k][j] *= autoprior[j] / autoprior[k]
                    S[j][k] = 1.0 / S[k][j]
                else:
                    S[k][j] = 0.0
                    S[j][k] = 0.0
        else:
            for j in range(max_order + 1):
                S[k][j] = 0.0
                S[j][k] = 0.0
    for k in range(max_order + 1):
        # cumulative distribution functions, by summing up the probabilities up to specific order
        if k == 0:
            kcdf[k] = 0.0
        else:
            kcdf[k] = kcdf[k - 1]
        # probability of specific order for the curve model
        pk[k] = 0.0
        for j in range(max_order + 1):
            pk[k] += S[j][k]
        if pk[k] > 0.0:
            pk[k] = 1.0 / pk[k]
        kcdf[k] += pk[k]
    return 0