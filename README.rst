mtest
=====

`mtest` is a Python implementation of the m-test, a two-sample test
based on model selection and described in [1] and [2].

Despite their importance in supporting experimental conclusions, standard
statistical tests are often inadequate for research areas, like the life sciences,
where the typical sample size is small and the test assumptions difficult to
verify. In such conditions, standard tests tend to be overly conservative, and
fail thus to detect significant effects in the data.

The m-test is a classical statistical test in the sense of defining significance
with the conventional bound on Type I errors. On the other hand, it is based
on Bayesian model selection, and thus takes into account uncertainty about the
model’s parameters, mitigating the problem of small samples size.

The m-test has been found to generally have a higher power (smaller fraction of
Type II errors) than a t-test error for small sample sizes (3 to 100 samples).

[1] Berkes, P., Fiser, J. (2011) `A frequentist two-sample test based on Bayesian model selection. <http://arxiv.org/abs/1104.2826>`_ arXiv:1104.2826v1 

[2] Berkes, P., Orban, G., Lengyel, M., and Fiser, J. (2011). `Spontaneous cortical activity reveals hallmarks of an optimal internal model of the environment. <http://www.sciencemag.org/content/331/6013/83.abstract>`_ Science, 331:6013, 83–87.

mtest tables
============

`mtest` ships caches tables of statistics to compute the p-value and
power of new data in the most efficient way. The library is
distributed with tables for p-values (type I error) for N=3,4,...,20
and for N=30,40,...,100. These tables cover the most common cases. New
tables are computed when needed, although completion might take a few
hours. Type II error tables are not included to keep the package size
small.

See `scripts\compute_basic_tables.py` for an example script to
pre-compute tables you might need. The script makes use of the `joblib
<http://packages.python.org/joblib/>`_ library to distribute the
computations on multiple cores.

Dependencies
============

`mtest` requires `SciPy <http://www.scipy.org/>`_ and `PyMC <http://code.google.com/p/pymc/>`_.

License
=======

`mtest` is released under the GPL v3. See LICENSE.txt .

Copyright (c) 2011, Pietro Berkes. All rights reserved.
