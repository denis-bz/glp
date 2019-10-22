#!/usr/bin/env python
""" glp.py: gnu linear programming kit GLPK <-> numpy
    see glp.md under https://gist.github.com/denis-bz
"""
    # test: lp-randomc.py

from __future__ import division, print_function
import traceback
import numpy as np
from scipy import sparse

import glpk  # https://github.com/bradfordboyle/pyglpk
import glp

from glp.zutil import Bag, boundsvec
from numpy import inf

#...............................................................................
def gnulp_to_lp( gnulp, drop_uncon=True, verbose=1 ):
    """ gnulp .matrix .rows .cols .obj -> LP( A b c ... ) numpy arrays, A scipy.sparse
        from e.g. gnulp = glpk.LPX( gmp=filename.mod )  # gmp= cpxlp= freemps=
    """
    i, j, data = zip( *gnulp.matrix )
    A = sparse.coo_matrix( (data, (i, j) )).tocsr()
    blo, b = np.asarray([ row.bounds for row in gnulp.rows ]) .T  # always n x 2 ?
    lb, ub = np.asarray([ col.bounds for col in gnulp.cols ]) .T
    c = np.array( list( gnulp.obj ), dtype=float )
    name = gnulp.name

    blo = boundsvec( blo, len(blo), - inf )
    b = boundsvec( b, len(b), inf )  # brandy.mps.gz A 219, b 220 ??
    lb = boundsvec( lb, len(lb), 0 )
    ub = boundsvec( ub, len(ub), inf )
    rowname = np.array( [row.name for row in gnulp.rows] )  # np.array for savez ?
    colname = np.array( [col.name for col in gnulp.cols] )
    if drop_uncon:
        A, b, blo, rowname = drop_unconstrained_rows( A, b, blo, rowname, name, verbose=verbose )

    return glp.LP( A=A, b=b, c=c, blo=blo, lb=lb, ub=ub,
            problemname=name, rowname=rowname, colname=colname,
            maximize=gnulp.obj.maximize,
            verbose=verbose )


def drop_unconstrained_rows( A, b, blo, rowname, name, verbose=1 ):
    """ drop rows blo -inf <= Ax <= b inf """
    jfinite = np.isfinite( blo ) | np.isfinite( b )
    ncon = jfinite.sum()
    nuncon = len(b) - ncon
    if nuncon > 0:
        if verbose:
            print( "drop_unconstrained_rows: %d -> %d rows" % (len(b), ncon) )
        return A[jfinite], b[jfinite], blo[jfinite], rowname[jfinite]
    else:
        return A, b, blo, rowname


#...............................................................................
def lp_to_gnulp( lp, verbose=1 ):
    """ LP( A b c blo lb ub ... ) -> glpk .matrix .rows .cols ... """
    A, b, c, blo, lb, ub, problemname \
        = lp.A, lp.b, lp.c, lp.blo, lp.lb, lp.ub, lp.problemname
    nr, nc = A.shape
    b = boundsvec( b, nr, inf )
    blo = boundsvec( blo, nr, - inf )
    lb = boundsvec( lb, nc, 0 )
    ub = boundsvec( ub, nc, inf )

    def _infnone( x ):
        return x if np.isfinite(x) \
            else None  # grr

    glp = glpk.LPX()  # empty
    glp.name = problemname
    glp.rows.add( nr )
    glp.cols.add( nc )

    S = sparse.coo_matrix( A )
    Srow = S.row.tolist()  # not astype(int)
    Scol = S.col.tolist()
    glp.matrix = zip( Srow, Scol, S.data )  # py2 list of tuples, py3 generator

    for j, cj in enumerate( c ):
        glp.obj[j] = cj
    glp.obj.maximize = lp.maximize

    for row, bi, bloi in zip( glp.rows, b, blo ):
        row.bounds = _infnone( bi ) if bi == bloi \
            else _infnone( bloi ), _infnone( bi )
    if lp.rowname is not None:
        for row, nm in zip( glp.rows, lp.rowname ):
            row.name = nm

    for col, l, u in zip( glp.cols, lb, ub ):
        col.bounds = _infnone( l ) if l == u \
            else _infnone( l ), _infnone( u )
            # glp/13-powplant.glp  j 1 s -868
    if lp.colname is not None:
        for col, nm in zip( glp.cols, lp.colname ):
            col.name = nm

    if verbose:
        print( "\nlp_to_gnulp: A %s, %d non0 " % (
                A.shape, S.nnz ))

    return glp

#...............................................................................
def gnulp_solve( gnulp, solver="simplex", verbose=1 ):
    """ gnulp / LPX simplex() or interior() -> Bag( obj, x, y, status, info ) """
        # glpsol -h: 100 options
    nr = len(gnulp.rows)
    nc = len(gnulp.cols)
    minmax = "maximize" if gnulp.obj.maximize  else ""
    info = "%s %s  %d rows, %d cols  %s " % (  # nnz ?
            gnulp.name, solver, nr, nc, minmax )
    if verbose:
        print( "\n{ gnulp_solve", info )
    glpk.env.term_on = bool(verbose)  # grr several calls ?
    gnulp.scale()
    try:
        if solver.startswith(( "interior", "ip" )):
            gnulp.interior()
            status = gnulp.status_i
        else:
            gnulp.simplex()  # no per-iter ?
            status = gnulp.status_s
        status_dual = gnulp.status_dual  # ip: undef ?
        obj = gnulp.obj.value  # c.x
    except RuntimeError:
        # glp/dea.glp NUMERIC INSTABILITY; SEARCH TERMINATED
        # RuntimeError: bad internal state for last solver identifier, glpsol is ok
        # glpsol is ok ?? Perturbing LP to avoid stalling
        traceback.print_exc()
        raise

    x = np.array([ col.primal for col in gnulp.cols ], dtype=float )
    y = np.array([ row.dual for row in gnulp.rows ], dtype=float )
    # gap = obj - b.dot( y )  # no, glpk.pdf p. 52
    if verbose:
        print( "obj: %g  status: %s %s  %s" % (
                obj, status, status_dual, info ))
        print( "x: %.3g .. %.3g  %s" % (x.min(), x.max(), x))
        print( "y: %.3g .. %.3g  %s" % (y.min(), y.max(), y))
        print( "}\n" )
    elif gnulp.status == "unbnd":
        print( "Warning: gnulp_solve: unbounded ", info )

    return Bag( obj=obj, x=x, y=y, status=status, status_dual=status_dual, info=info )

