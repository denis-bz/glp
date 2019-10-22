#!/usr/bin/env python
""" Klee-Minty_cube: D variables, D constraints, 2^D vertices => slow simplex ?
    https://en.wikipedia.org/wiki/Klee-Minty_cube
"""
    # glpk simplex d=200 0.0 sec, final f == 5^d (uses trianguler ?)
    # run-linprog-Klee-Minty linprog d=20: 70 sec real
    #   final f 6.81e+13  should be 5^d = 9.54e+13

from __future__ import division, print_function
import numpy as np
from scipy import sparse as sp

#...............................................................................
def Klee_Minty_cube( d, inc=1, verbose=1 ):
    """ Klee-Minty_cube: D variables, D constraints, 2^D vertices => slow simplex ?
        https://en.wikipedia.org/wiki/Klee-Minty_cube
        -> A dense, b, c
    """
        # ill-conditioned:
        #   inc=0 d=20: sing 1.4e6 .. 7e-7
        #   inc=0 d=200: 2e60 .. 1.5e-16
        # todo: scale
    A = sp.eye( d, format="coo" )
    for k in range( 1, d ):
        A.setdiag( 2 ** (k + 1), k=- k )
    A = A.toarray()
    if inc:
        A += inc
    dpow = np.arange( d )
    b = 5. ** (dpow + 1)    # 5 25 .. 5^d
    c = - (2. ** dpow)[::-1]  # 2^(d-1) 2^(d-2) .. 1  minimize
    if verbose:
        sing = np.linalg.svd( A, compute_uv=False )
        print( """
Klee_Minty_cube( d=%d, inc=%g )
A:
%s
b: %s
c: %s
singular values: %s
""" % (d, inc, A, b, c, sing ))

    return A, b, c

#...............................................................................
if __name__ == "__main__":
    import sys

    np.set_printoptions( threshold=10, edgeitems=5, linewidth=120,
            formatter = dict( float = lambda x: "%.2g" % x ))  # float arrays %.2g

    d = 20
    inc = 1
    verbose = 1
    save = 0
    tag = "tmp"  # save > tag.lp

    # to change these params, run this.py a=1 b=None 'c = expr' ... in sh or ipython --
    for arg in sys.argv[1:]:
        exec( arg )

    #...............................................................................
    A, b, c = Klee_Minty_cube( d, inc=inc, verbose=verbose )

    if save:
        import glp  # https://gist.github.com/denis-bz glp.md
        name = "%d-iinc%g-Klee-Minty" % (d, iinc)
        lp = glp.LP( A, b, c, problemname=name )
        glp.save_lp( tag + ".lp", lp )  # .lp: cplex

