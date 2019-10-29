#!/usr/bin/env python
""" save_lp.py  28oct 2019 _write_amplmod """

from __future__ import division, print_function
import numpy as np
from scipy import sparse as sp

import glp
from glp.zutil import Bag
from glp.load_lp import _gnufiletype

#...............................................................................
def save_lp( outfile, lp, verbose=1 ):
    """ LP() or gnulp -> outfile .lp .mps ... """
    if outfile.endswith( ".mod" ):
        return _write_amplmod( outfile, lp, verbose=verbose )
    if isinstance( lp, Bag ):  # lprec
        lp = glp.lp_to_gnulp( lp )
    # assert isinstance( lp, glpk.LPX )
    lp.write( **_gnufiletype( outfile ))  # .lp .mps .glp
    return lp


def _write_amplmod( outfile, lp, header="", footer="end;", verbose=1 ):
    """ LP( A b c ... ) -> ampl .mod, v similar to .lp cplex """
        # a tiny subset of .mod, can ampl read .lp ?
    assert lp.rowname is not None
    assert lp.colname is not None
    f = open( outfile, "w" )
    if verbose:
        print( "write_modfile: %s  A %s nnz %d" % (
                outfile, lp.A.shape, lp.A.nnz) )

    def pr( *args ):
        print( *args, end="", file=f )

    if header:
        pr( header + "\n" )
    pr( "# problem %s;\n" % lp.problemname )  # not glpsol
    for varname, lb, ub in zip( lp.colname, lp.lb, lp.ub ):
        pr( "var %s >= %g" % (varname, lb) )  # "varname" ?
        if ub != np.inf:
            pr( ", <= %g" %  ub )
        pr( ";\n" )
    pr( "maximize" if lp.maximize else "minimize", " obj:" )  # lp.objectivename ?
    pr( _sumstr( lp.c, names=lp.colname ))  # sum c_j * x_j
    pr( " ;\n" )
    pr( "subject to \n" )
    A = lp.A.tocsr()

    for Arow, rowname, b, blo in zip(
            A, lp.rowname, lp.b, lp.blo ):
        pr( "%s: " % rowname )
        rowsum = _sumstr( Arow, names=lp.colname )  # sum Aij * xj
        if b == blo:
            pr( "%s = %g ;\n" % (rowsum, b) )
            continue
        if np.isfinite( b ):
            pr( "%s <= %g ;\n" % (rowsum, b) )
        if np.isfinite( blo ):
            if np.isfinite( b ):
                pr( "%s_: " % rowname )  # both: row <=  row_ >=
            pr( "%s >= %g ;\n" % (rowsum, blo) )
    if footer:
        pr( footer + "\n" )
    f.close()


def _sumstr( x, ix=None, names=None, near0=1e-10, fmt=" %+.6g*" ):
    """ [2, -3, 1, -1]
    -> " +2*x0 -3*x1 +x2 -x3"
    ix=[1, 3]: " -3*x1 -x3"
    """
    if sp.issparse( x ):
        x = x.tocsr()
        x, ix = x.data, x.indices
    if ix is None:
        ix = range( len(x) )
    terms = []
    for j, val in zip( ix, x ):
        if abs(val) <= near0:
            continue
        s = (" +" if val == 1 else
            " -" if val == -1 else
            fmt % val)  # +-val*name
        name = (names[j] if names is not None
            else "x%d" % j)
        terms.append( s + name)

    return "".join( terms )

#...............................................................................
if __name__ == "__main__":
    import os
    import sys
    from glp.load_lp import load_lp
    from glp.zutil import scan_args, globs

    np.set_printoptions( threshold=20, edgeitems=10, linewidth=140,
            formatter = dict( float = lambda x: "%.2g" % x ))  # float arrays %.2g
    print( "\n" + 80 * "=" )

    lpfiles = "../data/glp/01/04-plan.glp"  # plan.glp: double bound i 8 d 250 300
    # lpfiles = "../netlib/zib/mps/*.mps.gz"  # non-alph var names ?
    # lpfiles = "../netlib/zib/mps/brandy.mps.gz"  # A 219, b 220 ??  ship* too
    nin = 10
    save = 1
    to = "lp"  # glpk | lp | linprog

        # run my.py [a=1 b=None 'c = expr' ...] [file* ...] in shell or IPython
    eqargs, fileargs = scan_args( sys.argv )
    for eqarg in eqargs:
        exec( eqarg )

    for lpfile in fileargs \
            or globs( lpfiles )[:nin]:
        print( "\n" + 80 * "-" )
        print( "--", lpfile )

        lp = load_lp( lpfile, to=to, verbose=1 )  # Bag( A b c blo lb ub )  drop_uncon
        if to == "linprog":
            lp, linrec = lp  # grr
        if save:
            basename = os.path.splitext( os.path.basename( lpfile ))[0]
            gnulp = save_lp( basename + ".mod", lp )  # .mod: write_modfile

