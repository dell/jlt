#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''linearMapping.py'''

'''
MIT License

Copyright (c) 2023 Dell Technologies, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''
2023-04-02
Ben Fauber
Dell Technologies
Austin, Texas, USA
'''

import math
import numpy as np
import scipy.sparse as sp
import fht      #Fast hadamard transform from https://github.com/nbarbey/fht


def approximate_k_jlt(m, eps):
    '''
    Approximates `k` for d-to-k JL projection with defined `eps` value. Calculation
    is dependent on `m` (number of instances) only. NOT dependent on `d`.
    
    S. Dasgupta and A. Gupta. `An elementary proof of the Johnson-Lindenstrauss 
    Lemma.` 1999, http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.45.3654

    Parameters
    ----------
    m : int, >0,
        The number of row vectors in data set (i.e. number of instances).
    eps : float in [0,1].
        Maximum distortion rate as defined by the Johnson-Lindenstrauss lemma.
        
    Returns
    -------
    k : int, int
        Minimal number of k-dimensions to guarantee, 
        with good probability, an eps value from d-dimensions.
    '''
    denominator = (math.pow(eps,2) / 2) - (math.pow(eps,3) / 3)
    return int((4 * math.log(m)) / denominator)


def calculate_q(d, n, eps, p):
    '''
    Calculates the variance `q` of the Gaussian normal 
    distriubtion N(0,q^(-1)) for the k x d matrix P of the FJLT.

    Parameters
    ----------
    d : int greater than 0,
        Number of components in the vector.
    n : int greater than 0.
        Number of all samples (i.e. k x d).
    eps : float or numpy array of float in [0,1].
        Maximum distortion rate as defined by the Johnson-Lindenstrauss lemma.
        If an array is given, it will compute a safe number of components
        array-wise.
    p : int of {1,2},
        The type of embedding desired: L1 or L2, 
        therefore p = 1 or 2, respectively.

    Returns
    -------
    q : float greater than 0,
        Variance of the of the Gaussian normal distriubtion 
        N(0,q^(-1)) for the k x d matrix P of the FJLT (Rho = PHD).    
    '''
    q_calc = (math.pow(eps,(p-2))*(math.pow((math.log(n)),p)))/d
    return min([q_calc, 1])


def nextPow(d_act):
    '''
    Helper function.
    Used to pad the signal prior to passing to FFT. Doing so 
    can speed up the computation of the FFT when the signal length is 
    not an exact power of 2.
    
    Returns the exponents for the smallest powers of two that satisfy
    2^p ≥ |d_act|
    for each element in `d_act`.
    
    Parameters
    ----------
    d_act : int greater than 0,
        
    Returns
    -------
    P : int,
        Exponent for the smallest power of two that satisfies:
        2^p ≥ |d_act|
        for each element in `d_act`.
    '''
    d_act = d_act - 1
    d_act |= d_act >> 1
    d_act |= d_act >> 2
    d_act |= d_act >> 4
    d_act |= d_act >> 8
    d_act |= d_act >> 16
    d_act += 1
    return d_act


def random_sparse_Rademacher_matrix(d, k, s, density, random_seed, swr):
    '''
    Helper function to generate a sparse random Rademacher csr-format
    matrix of defined sparsity/density. Non-zero values are
    sampled from a random Rademacher distribution.
    
    Parameters
    ----------
    d : int > 0
        matrix size
    k : int > 0
        matrix size
    s : float
        Rademacher matrix scalar
    density : float [0,1]
        density/sparsity of the matrix. Density=0 is a matrix with NO
        non-zero elements.
    random_seed : int
        random seed value for the random normal distrubtion sampling.
    swr : Boolean (default=False)
        sample with replacement
    
    Returns
    -------
    matrix : k x d dimension sparse csr-format matrix
    ''' 
    rng1 = np.random.default_rng(seed=random_seed)
    rng2 = np.random.default_rng(seed=random_seed+23)  
    indices = []
    offset = 0
    indptr = [offset]
    for _ in range(k):
        # find the indices of the non-zero components for row i
        n_nonzero_i = rng1.binomial(d, density)
        indices_i = rng2.choice(d, size=n_nonzero_i, replace=swr)
        indices.append(indices_i)
        offset += n_nonzero_i
        indptr.append(offset)
    indices = np.concatenate(indices)
    scalar_ = math.sqrt(1 / density) / math.sqrt(k)
    two_scalar_ = 2 * scalar_
    data = np.multiply(rng2.binomial(1, 0.5, size=np.size(indices)), two_scalar_) - scalar_
    return sp.csr_matrix((data, indices, indptr), shape=(k, d))


def random_sparse_matrix(k, d, density, sd, random_seed, swr):
    '''
    Helper function to generate a sparse random csr-format
    matrix of defined sparsity/density. Non-zero values are
    sampled without replacement from a random normal distribution
    of mean=0 and variance=variance.
    
    Parameters
    ----------
    k : int > 0
        matrix size
    d : int > 0
        matrix size
    density : float [0,1]
        density/sparsity of the matrix. Density=0 is a matrix with NO
        non-zero elements.
    sd : float
        standard deviation of the random normal distribution from which the 
        non-zero values are sampled without replacement (SWOR).
    random_seed : int
        random seed value for the random normal distrubtion sampling.
    swr : Boolean (default=True)
        sample with replacement
    
    Returns
    -------
    matrix : k x d dimension sparse csr-formatted matrix
    '''
    rng1 = np.random.default_rng(seed=random_seed)
    rng2 = np.random.default_rng(seed=random_seed+23)

    nnz = int(k*d*density)
    row = rng1.choice(k, size=nnz, replace=swr)
    cols = rng2.choice(d, size=nnz, replace=swr)
    data = rng1.normal(0, sd, nnz)
    return sp.csr_matrix((data, (row, cols)), shape=(k, d))


def calculate_H(A, d, m, d_act):
    '''
    Helper function to calculates the H (Hadamard-Walsh) matrix
    of the FJLT. Input values are defined under the FJLT function.
    '''
    #Calculate the next power of 2
    m_act = nextPow(m)

    #Calculate H
    A_aug = np.zeros((d_act, m_act))
    A_aug[0:d, 0:m] = A
    H = fht.fht(A_aug)
    del A_aug
    return H[0:d, 0:m]


def calculate_D(d, k, d_act, random_seed):
    '''
    Helper function to calculates the D matrix of the FJLT. 
    Input values are defined under the FJLT function.
    '''
    #Calculate D
    rng = np.random.default_rng(seed=random_seed)
    sc_ft = math.sqrt(d_act / (d * k))
    sc_ft_prod = 2 * sc_ft
    D_diag = np.multiply(rng.integers(0, 2, size=d),sc_ft_prod) - sc_ft
    return sp.diags(D_diag, format='csr')


def calculate_D_SRHT(d, k, random_seed):
    '''
    Helper function to calculates the D matrix of the SRHT. 
    Input values are defined under the SRHT function.
    '''
    #Calculate D
    rng = np.random.default_rng(seed=random_seed)
    sc_ft = math.sqrt(d/k)
    sc_ft_prod = 2 * sc_ft
    D_diag = np.multiply(rng.integers(0, 2, size=d), sc_ft_prod) - sc_ft
    return sp.diags(D_diag, format='csr')



def calculate_P(d, k, q, random_seed, swr):
    '''
    Helper function to calculates the P matrix of the FJLT. 
    Input values are defined under the FJLT function.
    '''
    sd_ = math.sqrt(1/q)
    return random_sparse_matrix(k, d, q, sd_, random_seed, swr)


def random_selector(HDA, d, k, random_seed, swr):
    '''
    Helper function to generate a subset of HDA matrix.
    
    Parameters
    ----------
    HDA : array of d x m dimensions
        product of HD transform on A (follows FJLT matrix naming paradigm)
    k : int > 0
        desired output matrix size
    random_seed : int
        random seed value for the random normal distrubtion sampling.
    swr : Boolean (default=True)
        sample without replacement
    
    Returns
    -------
    matrix : k x m dimension matrix
    '''
    rng = np.random.default_rng(seed=random_seed+23)
    selected = rng.choice(d, size=k, replace=swr)
    return HDA[selected]


def calculate_R(d, k, s, random_seed, swr):
    '''
    Helper function to calculates the R-random matrix of Random Projections. 
    Input values are defined under the RP definitions.
    '''
    density = 1/s
    return random_sparse_Rademacher_matrix(d, k, s, density, random_seed, swr)


def JLT(A, d, k, random_seed, swr):
    '''
    Helper function to calculates a JLT Random Projection.

    W. B. Johnson and J. Lindenstrauss, `Extensions of Lipschitz mappings 
    into a Hilbert Space.` Contemp. Math. 1984, 26, 189-206.
    '''
    s = 1
        
    #Calculate R
    R = calculate_R(d, k, s, random_seed, swr)
    
    #Final product
    return R@A #sparse matrix multiplication is fast


def sparseRP(A, d, k, random_seed, swr):
    '''
    Helper function to calculates a sparse Random Projection.

    Dimitris Achlioptas, `Database-friendly random projections: Johnson-Lindenstrauss
    with binary coins.` J. Comput. Syst. Sci. 2003, 66(4), 671-687.
    '''
    s = 3
        
    #Calculate R
    R = calculate_R(d, k, s, random_seed, swr)
    
    #Final product
    return R@A #sparse matrix multiplication is fast


def verySparseRP(A, d, k, random_seed, swr):
    '''
    Helper function to calculates a sparse Random Projection.
    
    Peng, L.; Hastie, T. J.; Church, K. W. `Very Sparse Random Projections.` 
    KDD 2006, Proceedings of the 12th ACM SIGKDD international conference on 
    Knowledge discovery and data mining, August 2006, pages 287–296.
    '''
    s = math.sqrt(d)
    
    #Calculate R
    R = calculate_R(d, k, s, random_seed, swr)
    
    #Final product
    return R@A #sparse matrix multiplication is fast


def FJLT(A, d, k, m, eps, p, random_seed, swr):
    '''
    Helper function to calculate the FJLT.
    
    N. Ailon and B. Chazelle, `Approximate Nearest Neighbors and the Fast
    Johnson-Lindenstrauss Transform.` STOC’06, May21–23, 2006, 
    Seattle, Washington, USA. 
    and
    N. Ailon and B. Chazelle, `The Fast Johnson–Lindenstrauss Transform
    and Approximate Nearest Neighbors.` Siam. J. Comput. 2009, Vol. 39, 
    No. 1, pp. 302–322.
    '''
    n = d * m
    d_act = nextPow(d)

    #Calculate D
    D = calculate_D(d, k, d_act, random_seed)

    #Calculate P
    q = calculate_q(d, n, eps, p)
    P = calculate_P(d, k, q, random_seed, swr)

    #Final product
    DA = D@A #O(d) time complexity
    HDA = calculate_H(DA, d, m, d_act) #O(d log d) time complexity
    return P@HDA #sparse matrix multiplication is fast


def SRHT(A, d, k, m, random_seed, swr):
    '''
    Helper function to calculate the SRHT JLT.
    
    F. Krahmer and R. Ward, `New and improved Johnson-Lindenstrauss 
    embeddings via the Restricted Isometry Property.`
    SIAM J. Math. Anal. 2011, 43(3), 1269–1281. 
    '''
    d_act = nextPow(d)
    
    #Calculate D
    D = calculate_D_SRHT(d, k, random_seed)
    
    #Final product
    DA = D@A #O(d) time complexity
    HDA = calculate_H(DA, d, m, d_act) #O(d log d) time complexity
    return random_selector(HDA, d, k, random_seed, swr)


def linearMapping(A, k=None, eps=0.1, method='FJLT', p=2, random_seed=21):
    '''
    Dimensionality reduction through random projection with theoretical and 
    provable error (eps) guarantees. Transforms column vector or array of 
    d x m dimensions into k x m dimensions.

    NOTE: method will IGNORE the Johnson-Lindenstrauss lemma theoretical 
    guarantees if k != `None`
    
    W. B. Johnson and J. Lindenstrauss, `Extensions of Lipschitz mappings 
    into a Hilbert Space.` Contemp. Math. 1984, 26, 189-206. 
    and
    Dimitris Achlioptas, `Database-friendly random projections: Johnson-Lindenstrauss
    with binary coins.` J. Comput. Syst. Sci. 2003, 66(4), 671-687.
    and
    L. Peng, T. J. Hastie, K. W. Church, `Very Sparse Random Projections.` 
    KDD 2006, Proceedings of the 12th ACM SIGKDD international conference on 
    Knowledge discovery and data mining, August 2006, pages 287–296.
    and
    N. Ailon and B. Chazelle, `Approximate Nearest Neighbors and the Fast
    Johnson-Lindenstrauss Transform.` STOC’06, May21–23, 2006, 
    Seattle, Washington, USA. 
    and
    N. Ailon and B. Chazelle, `The Fast Johnson–Lindenstrauss Transform
    and Approximate Nearest Neighbors.` Siam. J. Comput. 2009, Vol. 39, 
    No. 1, pp. 302–322.
    and
    F. Krahmer and R. Ward, `New and improved Johnson-Lindenstrauss 
    embeddings via the Restricted Isometry Property.`
    SIAM J. Math. Anal. 2011, 43(3), 1269–1281. 
   
    Parameters
    ----------
    A : vector or array,
        Input vector or array of d x m dimensions.
    k : int
        Size of down-sampled dimension output required (d-to-k dimensions, where d > k).
    eps : float or numpy array of float in [0,1], {default = 0.1}
        Maximum distortion rate as defined by the Johnson-Lindenstrauss lemma.
    method : str, default = `FJLT`
        Options: `JLT`, `SparseRP`, `VerySparseRP`, `FJLT`, `SRHT`
    p : int of {1,2}, {default=2}
        The type of embedding desired: L1 or L2, 
        therefore p=1 or 2, respectively.
    random_seed : int
        Random seed value for the random number generator.
        
    Returns
    -------
    Phi : column vector or array,
        Output column vector or array of k x m dimensions.
    '''
    # check if input is vector (1-d) or matrix (2-d)
    if A.ndim == 1:
        # vector 
        d = A.shape[0]
        m = 1
        A = np.reshape(A, (d, m))
    else:
        # matrix
        d, m = A.shape
    
    eps = float(eps)

    if eps <= 0.0 or eps >= 1:
        raise ValueError(
            "The Method is defined for eps in [0, 1], got %r" % eps)
    if d <= 0:
        raise ValueError(
            "The Method is defined for d-dimensions greater than zero, got %r"
            % d)
    
    if k != None:
        k = int(k)
    else:
        k = approximate_k_jlt(m, eps)
    
    if method == 'JLT':
        # Method of Johnson and Lindenstrauss (1984)
        matrix_out = JLT(A, d, k, random_seed, swr=True)
    
    elif method == 'SparseRP':
        # Method of D. Achlioptas (2003)
        matrix_out = sparseRP(A, d, k, random_seed, swr=False)
        
    elif method == 'VerySparseRP':
        # Method of Peng, Hastie, and Church (2006)
        matrix_out = verySparseRP(A, d, k, random_seed, swr=False)
        
    elif method == 'FJLT':
        # Method of Ailon and Chazelle (2006)
        matrix_out = FJLT(A, d, k, m, eps, p, random_seed, swr=True)
    
    elif method == 'SRHT':
        # Method of Krahmer and Ward (2011)
        matrix_out = SRHT(A, d, k, m, random_seed, swr=True)

    else:
        raise ValueError(
            "The Method is defined for linear projections: `JLT`, `SparseRP`, `VerySparseRP`, `FJLT`, and `SRHT`, got %r"
            % method)
            
    return matrix_out



########################################################################


def _get_rffs(X, k, gamma, return_vars, random_seed):
    """
    Helper function to return random Fourier features based on data X (column vectors)
    and random variables W and b.
    
    Input:
    ------
    X = d-dimensional column vector (1, d) or column matrix (n, d): array
    k = dimension size to project down to : int
    gamma = gamma^2 is the variance
    return_vars = return W and b : Boolean
    random_seed = random seed for selection from Gaussian : int
        
    Returns:
    --------
    Z = random Fourier features for X, of dimensions (k, n) : array
    """
    if X.ndim == 1:
        # vector 
        d = X.shape[1]
        n = 1
        X = np.reshape(X, (n, -1))
    else:
        # matrix
        n, d = X.shape
    rng = np.random.default_rng(seed=random_seed)
    W = rng.normal(loc=0, scale=1, size=(k, d))
    b = rng.uniform(0, 2*math.pi, size=k)
    B = np.repeat(b[:, np.newaxis], n, axis=1)
    norm = 1./ math.sqrt(k)
    Z = norm * math.sqrt(2) * np.cos(gamma * W @ X.T + B)
    if return_vars:
        return Z, W, b
    else:
        return Z


def _get_srht_rffs(X, k, gamma, return_vars, random_seed):
    """
    Helper function to return random Fourier features based on data X (column vectors)
    and random variables W and b.
    
    Input:
    ------
    X = d-dimensional column vector (1, d) or column matrix (n, d): array
    k = dimension size to project down to : int
    gamma = gamma^2 is the variance
    return_vars = return W and b : Boolean
    random_seed = random seed for selection from Gaussian : int
        
    Returns:
    --------
    Z = random Fourier features for X, of dimensions (k, n) : array
    """
    if X.ndim == 1:
        # vector 
        d = X.shape[1]
        n = 1
        X = np.reshape(X, (n, -1))
    else:
        # matrix
        n, d = X.shape
    rng = np.random.default_rng(seed=random_seed)
    #W = rng.normal(loc=0, scale=1, size=(k, d))
    SRHT = linearMapping(X.T, k, eps=0.1, method='SRHT', p=2, random_seed=random_seed)
    b = rng.uniform(0, 2*math.pi, size=k)
    B = np.repeat(b[:, np.newaxis], n, axis=1)
    norm = 1./ math.sqrt(k)
    Z = norm * math.sqrt(2) * np.cos(gamma * SRHT + B)
    if return_vars:
        return Z, W, b
    else:
        return Z


def rff(X, k, gamma=1.0, random_seed=123):
    """
    Returns random Fourier features `Z` based on data `X` (column vectors).
    
    Input:
    ------
    X = d-dimensional column vector (1, d) or column matrix (n, d): array
    k = dimension size to project down to : int
    gamma = gamma^2 is the variance
    return_vars = return W and b : Boolean
    random_seed = random seed for selection from Gaussian : int
        
    Returns:
    --------
    Z = random Fourier features for X, of dimensions (k, n) : array
    """
    return_vars = False
    return _get_rffs(X, k, gamma, return_vars, random_seed)


def srht_rff(X, k, gamma=1.0, random_seed=123):
    """
    Returns random Fourier features `Z` based on data `X` (column vectors).
    
    Input:
    ------
    X = d-dimensional column vector (1, d) or column matrix (n, d): array
    k = dimension size to project down to : int
    gamma = gamma^2 is the variance
    return_vars = return W and b : Boolean
    random_seed = random seed for selection from Gaussian : int
        
    Returns:
    --------
    Z = random Fourier features for X, of dimensions (k, n) : array
    """
    return_vars = False
    return _get_srht_rffs(X, k, gamma, return_vars, random_seed)


def rbfMapping(A, k=None, method='SRHT-RFF', gamma=1.0, random_seed=21):
    '''
    Dimensionality reduction through Random Fourier Features with theoretical and 
    provable error (eps) guarantees. Transforms column vector or array of 
    d x m dimensions into k x m dimensions.
    
    A. Rahimi and B. Recht. `Random Features for Large-Scale Kernel Machines.`
    NeurIPS 2007.
    and
    F. Krahmer and R. Ward, `New and improved Johnson-Lindenstrauss 
    embeddings via the Restricted Isometry Property.`
    SIAM J. Math. Anal. 2011, 43(3), 1269–1281. 
    and
    Y. Cherapanamjeri and J. Nelson. `Uniform Approximations for Randomized 
    Hadamard Transforms with Applications.` March 3, 2022, 
    http://arxiv.org/abs/2203.01599v1
   
    Parameters
    ----------
    A : column vector or array,
        d-dimensional column vector (1, d) or column matrix (m, d): array
    k : int
        Size of down-sampled dimension output required (d-to-k dimensions).
    gamma : float {default = 1.0}
        gamma^2 is the variance of the Gaussian distribution of the sampling matrix.
    method : str, default = `SRHT-RFF`
        Options: `SRHT-RFF`, `RFF`
    random_seed : int
        Random seed value for the random number generator.
        
    Returns
    -------
    Phi : column vector or array,
        Output column vector or array of k x m dimensions.
    '''
    # check if input is vector (1-d) or matrix (2-d)
    if A.ndim == 1:
        # vector 
        d = A.shape[1]
        m = 1
        A = np.reshape(A, (m, -1))
    else:
        # matrix
        m, d = A.shape

    if d <= 0:
        raise ValueError(
            "The Method is defined for d-dimensions greater than zero, got %r"
            % d)
    
    if k != None:
        k = int(k)
    else:
        k = approximate_k_jlt(m, eps)
    
    if method == 'SRHT-RFF':
        # Method of XX and Nelson (2022)
        matrix_out = srht_rff(A, k, gamma, random_seed)
    
    elif method == 'RFF':
        # Method of Rahimi and Recht (2007)
        matrix_out = rff(A, k, gamma, random_seed)

    else:
        raise ValueError(
            "The Method is defined for Radial Basis Function projections: `RFF` and `SRHT-RFF`, got %r"
            % method)
            
    return matrix_out
