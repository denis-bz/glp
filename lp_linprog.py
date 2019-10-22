#!/usr/bin/env python
""" lp_linprog.py: LP( A b c ... ) <-> scipy linprog Bag( A_ub b_ub A_eq b_eq bounds )
"""

from __future__ import division, print_function
import numpy as np
from numpy import inf, all, any, isfinite
from scipy import sparse

import glp
from glp.zutil import Bag, inftonone

#...............................................................................
def lp_to_linprog( lp, verbose=1 ):
    """ in: LP( A b c ... ), see lprec.py
        ->lp, linrec:
            lp: LP() 
                (after rows_le, may change A b blo)
            linrec: Bag( A_ub b_ub A_eq b_eq bounds c )
                    for scipy linprog( **linrec )

        blo <= Ax <= inf: flip to - Ax <= - blo
        blo <= Ax <= b: add rows  ** change A b **
        split A b -> A_eq A_ub, b_eq b_ub  csr_matrix or None
        rownames, colnames ? ouch
    """
    A, b, c, blo, lb, ub, name \
        = lp.A, lp.b, lp.c, lp.blo, lp.lb, lp.ub, lp.problemname
    A = sparse.csr_matrix( A )  # csR, don't change
    # b blo lb ub +- inf, not None
    if lp.maximize:
        lp.c *= -1
        lp.maximize = False

        # change A b blo for linprog: rows_le, split_Ab
    lp.A, lp.b, lp.blo = rows_le( A, b, blo )

    linrec = split_Ab( lp.A, lp.b, lp.blo )  # grr A_ub b_ub A_eq b_eq
    linrec.bounds = np.c_[ inftonone(lb), inftonone(ub) ]
    linrec.c = c
    if verbose:
        print( "lp_to_linprog %s:" % name )
        _norms_near0( A, "|A| rows", axis=1 )
        # _norms_near0( A, "|A| cols", axis=0 )
        _print_linrec( name=name, **linrec )

    return lp, linrec  # Bag( A_ub b_ub A_eq b_eq bounds c ) for scipy linprog


def linprog_to_lp( linrec, name="noname" ):
    """ test-loopback.py: lp_to_linprog, linprog_to_lp """
    A_ub, b_ub, A_eq, b_eq, c \
        = linrec.A_ub, linrec.b_ub, linrec.A_eq, linrec.b_eq, linrec.c
    lb, ub = linrec.bounds.T  # grr inftonone

    if A_ub is None:  # all blo == Ax == b
        return glp.LP( A_eq, b_eq, c, blo=b_eq, lb=lb, ub=ub, problemname=name )

    if A_eq is None:
        blo = - inf * np.ones( len(b_ub ))  # all -inf <= Ax <= b
        return glp.LP( A_ub, b_ub, c, blo=blo, lb=lb, ub=ub, problemname=name )

    A = sparse.vstack(( A_ub.tocsr(), A_eq.tocsr() ))
    b = np.hstack(( b_ub, b_eq ))
    blo = np.r_[ - inf * np.ones( len(b_ub )), b_eq ]

    return glp.LP( A, b, c, blo=blo, lb=lb, ub=ub, problemname=name )  # maximize=False


#...............................................................................
def rows_le( A, b, blo ):
    """ -inf <= Ax <= inf: drop these rows, unconstrained
        finite <= Ax <= inf: flip to - Ax <= - blo
        2-sided blo < b, both finite: add rows - Ax <= - blo
        blo == b: asis
    """
        # -inf <= Ax <= inf, unconstrained
        # (drop_unconstrained_rows in glp.py too)
    juncon = (blo == -inf) & (b == inf)
    nuncon = juncon.sum()
    if nuncon > 0:
        jcon = ~ juncon
        ncon = jcon.sum()
        print( "dropping unconstrained rows: %d -> %d" % (len(b), ncon) )
        A, b, blo = A[jcon], b[jcon], blo[jcon]

        # finite <= Ax <= inf: flip to - Ax <= - blo
    jflip = isfinite( blo ) & (b == inf)
    if any( jflip ):
        A[jflip] *= -1
        b[jflip] = - blo[jflip]
        blo[jflip] = - inf
        # print( "rows_le jflip: %s \n%s \n%s " % (jflip.astype(int), blo, b ))

        # 2-sided blo <= Ax <= b, both finite: add rows - Ax <= - blo
    jboth = isfinite( blo ) & (blo < b) & isfinite( b )  # leave blo == b alone
    if any( jboth ):
        A = A.tocsr()
        A = sparse.vstack(( A, - A[jboth] ))
        b = np.hstack(( b, - blo[jboth] ))
        blo[jboth] = - inf
        blo = np.hstack(( blo, - inf * np.ones( jboth.sum() )))
        # print( "rows_le jboth: %s \n%s \n%s " % (jboth.astype(int), blo, b ))

    assert A.shape[0] == len(b) == len(blo), [A.shape[0], len(b), len(blo)]
    assert all( (blo == -inf) | (blo == b) ), "\n%s \n%s" % (blo, b)
    return A, b, blo

#...............................................................................
def split_Ab( A, b, blo ):
    """ -> Bag( A_ub b_ub A_eq b_eq ) """
        # grr, stupid back-and-forth
    jeq = (b == blo)
    if not any( jeq ):
        return Bag( A_ub=A, b_ub=b,
                    A_eq=None, b_eq=None )
    if all( jeq ):
        return Bag( A_ub=None, b_ub=None,
                    A_eq=A, b_eq=b )
    A = A.tocsr()  # see SO vstack
    jne = (~ jeq).nonzero()[0]
    jeq = jeq.nonzero()[0]
    return Bag( A_ub=A[jne], b_ub=b[jne],
                A_eq=A[jeq], b_eq=b[jeq] )


def _norms_near0( A, text, axis=1, near=1e-8 ):
    from scipy.sparse.linalg import norm
    norms = np.sort( norm( A, axis=axis, ord=np.inf ))
    j = np.searchsorted( norms, near )
    if j > 0:
        print( "%s: %d of %d are ~ 0, then %s .. %s " % (
            text, j, len(norms), norms[j:j+3], norms[-3:] ))

def _print_linrec( A_ub, b_ub, A_eq, b_eq, name, **kw ):
    if A_ub is not None:
        print( "A_ub %s  b_ub %.3g .. %.3g  " % (
                A_ub.shape, b_ub.min(), b_ub.max() ))
    if A_eq is not None:
        print( "A_eq %s  b_eq %.3g .. %.3g " % (
                A_eq.shape, b_eq.min(), b_eq.max() ))
    print( "" )

