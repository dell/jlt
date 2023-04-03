# jlt

Johnson-Lindenstrauss transforms, random projections, and randomized Hadamard transforms in python 3.x

Supports linear mappings and radial basis function (rbf) mappings (a.k.a. Random Fourier Features) that reduce dimensionality while preserving the square of the $\ell2-norm$ between points with bounded error.

Created by:
[Ben Fauber](https://github.com/benfauber), Dell Technologies, 02Apr2023

***

## Overview

Provides python 3.x functions based on the Johnson-Lindenstrauss (JL) lemma. The Johnson-Lindenstrauss Transform (JLT) preserves pair-wise distances with bounded error $\epsilon$ as points are projected from high-dimensional space $d$ into a lower-dimensional space $k$. The functions in this package accepts $d$-dimensional vector and matrix/array inputs to return the $k$-dimensional output. The JLT preserves the square of the $\ell2-norm$ between points with bounded error $\epsilon$.

<P align="center">
<IMG SRC="/assets/jl_lemma.PNG" HEIGHT="60%" WIDTH="60%" CLASS="center" ALT="johnson-lindenstrauss lemma">
  </P>
  <P>

The figures below illustrate: 1) JLT algorithm runtimes; and 2) preservation of the square of the $\ell2-norm$ by the Fast JLT (FJLT). Random projections (RP) and Fast Johnson-Lindenstrauss Transform (FJLT) are faster versions of the original JLT, and subsampled randomized Hadamard transforms (SRHT) are even faster yet (first figure, gold line). The FJLT preserves the square of the $\ell2-norm$ regardless of the sparsity of the input (second figure). In both of the figures, $d$ is held constant at $d = 16,384$ and $k$ is varied (x-axis).

<IMG SRC="/assets/jlt_runtimes.png" HEIGHT="40%" WIDTH="40%" CLASS="center" ALT="johnson-lindenstrauss algorithm runtimes">
<IMG SRC="/assets/fjlt_l2normpreservation.png" HEIGHT="40%" WIDTH="40%" CLASS="center" ALT="fast johnson-lindenstrauss transform (FJLT) preservation of L2-norm">
<P>

JLT has applications in [linear mappings](https://en.wikipedia.org/wiki/Linear_map), [random projections](https://en.wikipedia.org/wiki/Random_projection), [locality-sensitive hashing LSH](https://en.wikipedia.org/wiki/Locality-sensitive_hashing), [matrix sketching](https://arxiv.org/abs/1206.0594), and [sparse recovery](https://www.cs.utexas.edu/~ecprice/courses/sublinear/bwca-sparse-recovery.pdf).

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
[in]> from linearMapping import linearMapping
```

***

## Functions

### linearMapping()
Produces linear mapping of input vector or array from `d` dimensions into `k` dimensions, typically applied where $d >> k$. Provides bounded guarantees of Johnson-Lindenstrauss lemma when `k` is determined automatically (via method of Dasgupta et al.) with user-defined `eps` ($\epsilon$ in Johnson-Lindenstrauss lemma) as the error associated with the preservation of the $\ell2-norm$.

`A` is the input vector $A \in \mathbb{R}^{d}$ or matrix $A \in \mathbb{R}^{n \times d}$. `p` is the $l{p}-norm$ where $p \in \{ 1,2 \}$, and is only relevant to the `FJLT` method.

Function accepts one of several methods: `JLT`, `SparseRP`, `VerySparseRP`, `FJLT`, or `SRHT`.

Defaults are: `k=None`, `eps=0.1`, `method=FJLT`, `p=2`, and `random_seed=21`. Code is fully commented -- variables and helper functions are further defined within the PY file.

```python
[in]> linearMapping(A, k=None, eps=0.1, method='FJLT', p=2, random_seed=21)
[out]> # d-to-k mapping of A
```

***

### References

W. B. Johnson and J. Lindenstrauss, "Extensions of Lipschitz mappings into a Hilbert Space." Contemp. Math. 1984, 26, 189-206. 

Dimitris Achlioptas, "Database-friendly random projections: Johnson-Lindenstrauss with binary coins." J. Comput. Syst. Sci. 2003, 66(4), 671-687. [link to paper](https://www.sciencedirect.com/science/article/pii/S0022000003000254)

L. Peng, T. J. Hastie, K. W. Church, "Very sparse random projections." KDD 2006, Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining, August 2006, pages 287–296. [link to paper](https://dl.acm.org/doi/10.1145/1150402.1150436)

N. Ailon and B. Chazelle, "Approximate Nearest Neighbors and the Fast Johnson-Lindenstrauss Transform." STOC’06, May21–23, 2006, Seattle, Washington, USA. [link to paper](http://www.cs.technion.ac.il/~nailon/fjlt.pdf)

F. Krahmer and R. Ward, "New and improved Johnson-Lindenstrauss embeddings via the restricted isometry property." SIAM J. Math. Anal. 2011, 43(3), 1269–1281. [link to paper](https://arxiv.org/abs/1009.0744)

N. Ailon and E. Liberty, "Almost Optimal Unrestricted Fast Johnson-Lindenstrauss Transform." ACM Trans. Algorithms 2013, 9(3), 1–12. [link to paper](https://arxiv.org/abs/1005.5513)

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
