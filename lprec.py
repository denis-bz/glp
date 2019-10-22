#!/usr/bin/env python
""" LP: linear-programming problem Bag( A b c blo lb ub ... )
    numpy arrays, A scipy.sparse
"""

from __future__ import division, print_function
import numpy as np
from scipy import sparse

from glp.zutil import Bag, boundsvec, nbytes, quantiles
from numpy import inf

#...........................................................................
def LP( A, b, c=0, blo=-inf, lb=0, ub=inf, problemname="noname",
        maximize=False, rowname=None, colname=None, mipvars=None,  # glpk
        verbose=1 ):
    """ -> Bag( A b c blo lb ub ... A sparse, b c ... numpy vecs """
    if not sparse.issparse( A ):
        A = sparse.csr_matrix( A )  # beware, hstack vstack may -> coo
    nr, nc = A.shape
    c = boundsvec( c, nc, 0 )  # zeros(nc) / array of len nc
    b = boundsvec( b, nr, inf )
    blo = boundsvec( blo, nr, - inf )
    lb = boundsvec( lb, nc, 0 )
    ub = boundsvec( ub, nc, inf )
    nnz = A.nnz if hasattr( A, "nnz" ) \
        else np.count_nonzero( A )
    lp = Bag(
        A=A,
        b=b,
        c=c,
        blo=blo,
        lb=lb,
        ub=ub,
        nnz=nnz,
        problemname=problemname,
            # glpk --
        maximize=maximize,
        rowname=rowname,
        colname=colname,
        mipvars=mipvars,
        )
    if verbose:
        print_lp( lp, verbose=verbose )
    return lp


def lp_check( lp, x ):
    """ print x c*x, check blo <= Ax <= b and lb <= x <= ub """
    A, b, c, blo, lb, ub \
        = lp.A, lp.b, lp.c, lp.blo, lp.lb, lp.ub
    print( "x  ", quantiles( x ))
    print( "c*x", quantiles( c * x ))
    Ax = A * x
    _le( Ax, b, "Ax", "b" )
    _le( blo, Ax, "blo", "Ax" )
    _le( lb, x, "lb", "x" )
    _le( x, ub, "x", "ub" )

def _le( xlo, xhi, lo, hi ):
        # check Ax <= b etc.
    j = (xlo - xhi).argmax()
    if xlo[j] > xhi[j] \
    and not np.isclose( xlo[j], xhi[j] ):
        print( "lp_check: %s > %s  %g > %g  at [%d] " % (
                lo, hi, xlo[j], xhi[j], j ))


def print_lp( lp, verbose=1, header="", footer="" ):
    """ print c A ... with caller's printoptions """
    A, b, c, blo, lb, ub, nnz, problemname \
        = lp.A, lp.b, lp.c, lp.blo, lp.lb, lp.ub, lp.nnz, lp.problemname
    minmax = "maximize" if lp.maximize  else "minimize"
    c0 = (1 - np.count_nonzero( c ) / len(c)) * 100
    c0 = "%.0f %% == 0 " % c0  if c0 >= 50 \
        else ""
    neq = (b == blo).sum()
    nlt = (np.isfinite(blo) & (blo < b)).sum()
    print(
"""
{ LP %s  %s
c  : %.3g .. %.3g  %s  %s
A  : %s  %d non0, %.2g mbytes, %.3g .. %.3g  
lb : %.3g .. %.3g  %s
ub : %.3g .. %.3g  %s
b  : %.3g .. %.3g  %s
blo : %d == b, %d < b  %.3g .. %.3g  %s\
""" % (
        header or problemname, minmax,
        c.min(), c.max(), c0, c,
        A.shape, A.nnz, nbytes(A) / 1e6, A.data.min(), A.data.max(),
        lb.min(), lb.max(), lb,
        ub.min(), ub.max(), ub,
        b.min(), b.max(), b,
        neq, nlt, blo.min(), blo.max(), blo
        ))

    if verbose >= 2 and A.nnz < 500:
        print( "A: \n", A.A )  # dense
    if lp.mipvars is not None:
        print( "mip: %d continuous  %d int  %d binary " % (
            (lp.mipvars == 'c').sum(),
            (lp.mipvars == 'i').sum(),
            (lp.mipvars == 'b').sum() ))
    if footer:
        print( footer )
    print( "}\n" )

