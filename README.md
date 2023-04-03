# jlt

Johnson-Lindenstrauss transform (JLT), random projection (RP), fast Johnson-Lindenstrauss transform (FJLT), and randomized Hadamard transform (RHT) in python 3.x

Supports linear mappings and radial basis function (RBF) mappings (a.k.a. Random Fourier Features) that reduce dimensionality while preserving the square of the $\ell_2$-norm between points with bounded error.

Created by:
[Ben Fauber](https://github.com/benfauber), Dell Technologies, 02Apr2023

***

## Overview

Provides python 3.x functions based on the Johnson-Lindenstrauss (JL) lemma. The Johnson-Lindenstrauss Transform (JLT) preserves pair-wise distances with bounded error $\epsilon$ as points are projected from high-dimensional space $d$ into a lower-dimensional space $k$. The functions in this package accept $d$-dimensional vector and/or matrix/array inputs to return the $k$-dimensional output. The JLT preserves the square of the $\ell_2$-norm between points with bounded error $\epsilon$.

At a high level, the Johnson-Lindenstrauss transform (JLT) is a dimensionality-reduction technique as illustrated below, where $n > 0$ and typically $d >> k$.

<P align="center">
<IMG SRC="/assets/overview.PNG" HEIGHT="40%" WIDTH="40%" CLASS="center" ALT="illustration of johnson-lindenstrauss lemma in practice">
  </P>
  <P>

Specifically, a Johnson-Lindenstrauss transform (JLT) $\Phi$ is a random linear map for any set $Z$ of $n$-points in $d$-dimensions, defined by a matrix $B \in \mathbb{R}^{k \times d}$, where $\epsilon \in (0,1]$ and the pair-wise Euclidean distance between points $u$ and $v$, $\forall (u,v) \in Z$, is defined by $(1 - \epsilon)\|\|u-v\|\|^2_{\ell_2} \le \|\|Bu-Bv\|\|^2_{\ell_2} \le (1 + \epsilon)\|\|u-v\|\|^2_{\ell_2}$.

The above equation can be further simplified where $B$ is replaced with the linear map $\Phi$ and $x = u - v$ such that $\|\|\Phi x\|\|^2_{\ell_2} = (1 \pm \epsilon)\|\|x\|\|^2_{\ell_2} \quad \forall x \in Z$.
    
The figures below illustrate: 1) JLT algorithm runtimes; and 2) preservation of the square of the $\ell_2$-norm by the Fast JLT (FJLT). Random projections (RP) and Fast Johnson-Lindenstrauss Transform (FJLT) are faster versions of the original JLT, and subsampled randomized Hadamard transforms (SRHT) are even faster yet (first figure, gold line). The FJLT preserves the square of the $\ell_2$-norm regardless of the sparsity of the input (second figure). In both of the figures, $d$ is held constant at $d$ = 16,384 and $k$ is varied (x-axis).

<P align="center">
<IMG SRC="/assets/jlt_runtimes.png" HEIGHT="40%" WIDTH="40%" CLASS="center" ALT="johnson-lindenstrauss algorithm runtimes">
<IMG SRC="/assets/fjlt_l2normpreservation.png" HEIGHT="40%" WIDTH="40%" CLASS="center" ALT="fast johnson-lindenstrauss transform (FJLT) preservation of L2-norm">
</P>
<P>

JLT has applications in [linear mappings](https://en.wikipedia.org/wiki/Linear_map), [random projections](https://en.wikipedia.org/wiki/Random_projection), [locality-sensitive hashing LSH](https://en.wikipedia.org/wiki/Locality-sensitive_hashing), [matrix sketching](https://arxiv.org/abs/1206.0594), [low-rank matrix approximations](https://en.wikipedia.org/wiki/Low-rank_matrix_approximations), and [sparse recovery](https://www.cs.utexas.edu/~ecprice/courses/sublinear/bwca-sparse-recovery.pdf).

***

## Dependencies and Installing

### Dependencies
Python 3.x packages `math`, `numpy`, `scipy.sparse`, and `fht` (https://github.com/nbarbey/fht)

### Installing
1) Clone the ```linearMapping.py``` python file to your working directory using either:

- **Python command line**
```python
git clone https://github.com/dell/jlt.git
```

or

- **Jupyter Notebook**
```python
import os, sys

path = os.getcwd()
os.chdir(path)

!git clone https://github.com/dell/jlt.git

sys.path.insert(0, path+'\jlt')
```

2) Import the module into your script:

```python
[in]> from linearMapping import linearMapping, rbfMapping
```

***

## Functions

### linearMapping()
Produces linear mapping of input vector or array from `d` dimensions into `k` dimensions, typically applied where $d >> k$. Provides bounded guarantees of Johnson-Lindenstrauss lemma when `k` is determined automatically (i.e., `k=None`), via the method of Dasgupta and Gupta, with user-defined `eps` ($\epsilon$ in Johnson-Lindenstrauss lemma) as the error associated with the preservation of the $\ell_2$-norm.

```python
[in]> linearMapping(A, k=None, eps=0.1, method='FJLT', p=2, random_seed=21)
[out]> # d-to-k linear mapping of A
```  

`A` is the input vector $A \in \mathbb{R}^{d}$ or matrix $A \in \mathbb{R}^{n \times d}$. 

`method` accepts one of several variants of the JLT: `JLT`, `SparseRP`, `VerySparseRP`, `FJLT`, or `SRHT`. See _References_ section below for more details on each method.
  
`p` is the $\ell{p}$-norm where $p \in \\\{ 1, 2 \\\}$ and is only relevant to the `FJLT` method.
  
`random_seed` is the random seed value for the generator function that randomizes the Gaussian and/or the row-selector function, based on the `method` employed.

Defaults are: `k=None`, `eps=0.1`, `method=FJLT`, `p=2`, and `random_seed=21`. Code is fully commented -- variables and helper functions are further defined within the PY file. 
  
The user can further edit the code to specify sampling with replacement `swr` or sampling without replacement `swor` for either faster or more accurate calculations, respectively. NOTE: `swor` is recommended when solving for inverse matrices with iterative solvers (e.g., compressed sensing applications).

### rbfMapping()
Produces radial basis function (RBF) mapping (a.k.a. Random Fourier Features) of input vector or array from `d` dimensions into `k` dimensions, typically applied where $d >> k$. Provides bounded guarantees of Johnson-Lindenstrauss lemma when `k` is determined automatically (i.e., `k=None`), via the method of Dasgupta and Gupta, with user-defined `eps` ($\epsilon$ in Johnson-Lindenstrauss lemma) as the error associated with the preservation of the $\ell_2$-norm.

```python
[in]> rbfMapping(A, k=None, method='SRHT-RFF', gamma=1.0, random_seed=21)
[out]> # d-to-k radial basis function mapping of A
```
 
`A` is the input vector $A \in \mathbb{R}^{d}$ or matrix $A \in \mathbb{R}^{n \times d}$. 

`method` accepts two variants of the RBF: `RFF` or `SRHT-RFF`. See _References_ section below for more details on each method.
  
`gamma` is the standard deviation of the Gaussian distribution.
  
`random_seed` is the random seed value for the generator function that randomizes the Gaussian and/or the row-selector function, based on the `method` employed.

Defaults are: `k=None`, `method=SRHT-RFF`, `gamma=1.0`, and `random_seed=21`. Code is fully commented -- variables and helper functions are further defined within the PY file. 

The user can further edit the code to specify sampling with replacement `swr` or sampling without replacement `swor` for either faster or more accurate calculations, respectively. NOTE: `swor` is recommended when solving for inverse matrices with iterative solvers (e.g., compressed sensing applications).
  
***

### References

`JLT` W. B. Johnson and J. Lindenstrauss, "Extensions of Lipschitz mappings into a Hilbert Space." Contemp. Math. 1984, 26, 189-206. [link to paper](http://stanford.edu/class/cs114/readings/JL-Johnson.pdf)

`SparseRP` Dimitris Achlioptas, "Database-friendly random projections: Johnson-Lindenstrauss with binary coins." J. Comput. Syst. Sci. 2003, 66(4), 671-687. [link to paper](https://www.sciencedirect.com/science/article/pii/S0022000003000254)

`VerySparseRP` L. Peng, T. J. Hastie, K. W. Church, "Very sparse random projections." KDD 2006, Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining, August 2006, pages 287–296. [link to paper](https://dl.acm.org/doi/10.1145/1150402.1150436)

`FJLT` N. Ailon and B. Chazelle, "Approximate Nearest Neighbors and the Fast Johnson-Lindenstrauss Transform." STOC’06, May21–23, 2006, Seattle, Washington, USA. [link to paper](http://www.cs.technion.ac.il/~nailon/fjlt.pdf)

`SRHT` F. Krahmer and R. Ward, "New and improved Johnson-Lindenstrauss embeddings via the restricted isometry property." SIAM J. Math. Anal. 2011, 43(3), 1269–1281. [link to paper](https://arxiv.org/abs/1009.0744)

`SRHT` N. Ailon and E. Liberty, "Almost Optimal Unrestricted Fast Johnson-Lindenstrauss Transform." ACM Trans. Algorithms 2013, 9(3), 1–12. [link to paper](https://arxiv.org/abs/1005.5513)

`RFF` A. Rahimi and B. Recht. "Random Features for Large-Scale Kernel Machines." NeurIPS 2007. [link to paper](https://papers.nips.cc/paper_files/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf)
  
`SRHT-RFF` Y. Cherapanamjeri and J. Nelson. "Uniform Approximations for Randomized Hadamard Transforms with Applications." 2022 Proceedings of the 54th Annual ACM SIGACT Symposium on Theory of Computing (STOC), 659–671. [link to paper](https://dl.acm.org/doi/abs/10.1145/3519935.3519961)
  
`k` S. Dasgupta and A. Gupta. "An elementary proof of the Johnson-Lindenstrauss Lemma." 1999. [link to paper](https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf)
  
Tight lower bounds for `k` K. G. Larsen and J. Nelson. "Optimality of the Johnson-Lindenstrauss Lemma." 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS). [link to paper](https://ieeexplore.ieee.org/document/8104096)
  
***

### Citing this Repo

```
@misc{FauberJLT2023,
  author = {Fauber, B. P.},
  title = {Johnson-Lindenstrauss Transforms},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dell/jlt}}
}
```
