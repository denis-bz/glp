#!/usr/bin/env python
""" load_lp( lpfile, to="glpk | lp | linprog" ) """
# 28oct 2019 _write_modfile

from __future__ import division, print_function
import os
import numpy as np

import glpk  # https://github.com/bradfordboyle/pyglpk
import glp
from glp.zutil import Bag

#...............................................................................
def load_lp( lpfile, to="lp", verbose=1 ):
    """ GLPK file .mod .mps ...  to= "lp" | "linprog" | "glpk"
        lp = load_lp( to="lp" ): Bag( A b c blo lb ub ... ), see LP()
        lp, linrec = load_lp( to="linprog" )
            linprog( **linrec ), a Bag( A_ub b_ub A_eq b_eq c bounds )
        gnulp = load_lp( to="glpk" ): a glpk rec, see pyglpk-brief.doc

        GLPK file formats: see glpk.pdf and http://lpsolve.sourceforge.net/5.5/formulate.htm
    """
    open( lpfile )  # else IOError: No such file or directory
    assert to in "glpk lp linprog ".split(), to

    gnulp = glpk.LPX( **_gnufiletype( lpfile ))  # read .mod .mps ...
        # free rows removed from .mps, others ?
    if to == "glpk":
        return gnulp  # .matrix .rows .cols .obj ...

    lp = glp.gnulp_to_lp( gnulp,  # Bag( A b c blo lb ub ... ), see lprec.py
            drop_uncon=True, verbose=verbose )
    if to == "lp":
        return lp  # LP( A b c ... ) after rows_le -- may change A b blo

        # linrec: Bag( A_ub b_ub A_eq b_eq c bounds ) for linprog( **linrec )
    lp, linrec = glp.lp_to_linprog( lp, verbose=verbose )
    if to == "linprog":
        return lp, linrec

    assert 0, to


#...............................................................................
def save_lp( outfile, lp, verbose=1 ):
    """ LP() or gnulp -> outfile .mod .mps ... """
    if outfile.endswith( ".mod" ):
        return _write_modfile( outfile, lp, verbose=verbose )
    if isinstance( lp, Bag ):  # lprec
        lp = glp.lp_to_gnulp( lp )
    # assert isinstance( lp, glpk.LPX )
    lp.write( **_gnufiletype( outfile ))  # .lp .mps .glp
    return lp


def _write_modfile( outfile, lp, header="", footer="end;", verbose=1 ):
    """ LP( A b c ... ) plain LP -> ampl .mod, similar to .lp
        todo: bounds
    """
        # https://www.gurobi.com/documentation/8.1/refman/lp_format.html#format:LP
    assert isinstance( lp, Bag ), type(lp).__name__  # LP( A b c ... )
    f = open( outfile, "w" )

    def pr( *args ):
        print( *args, file=f )

    if header:
        pr( "# %s\n" % header )
    pr( "maximize" if lp.maximize else "minimize", " obj:" )  # lp.objectivename ?
    pr( _sumstr( lp.c, names=lp.colname ))  # sum c_j * x_j
    pr( " ;\n" )
    pr( "subject to \n", file=f )
    A = lp.A.tocsr()

    for Arow, rowname, b, blo in zip(
            A, lp.rowname, lp.b, lp.blo ):
        pr( "# %s: %.3g .. .3%g \n" % (rowname, blo, b) )
        bfin = np.isfinite( b )
        blofin = np.isfinite( blo )
        if not( bfin and blofin ):
            continue
        pr( "%s: " % rowname )
        rowsum = _sumstr( Arow, names=lp.colname )  # sum Aij * xj
        if b == blo:
            pr( "%s == %g ;\n" % (rowsum, b) )
            continue
        if bfin:
            pr( "%s <= %g ;\n" % (rowsum, b) )
        if blofin:
            pr( "%s >= %g ;\n" % (rowsum, blo) )
    # todo: bounds
    if footer:
        pr( footer + "\n" )
    close( f )


def _sumstr( x, ix=None, names=None, near0=1e-10, fmt=" %+.6g*" ):
    """ [2, -3, 1, -1]
    -> " +2*x0 -3*x1 +x2 -x3"
    ix=[1, 3]: " -3*x1 -x3"
    """
    if sp.issparse( x ):
        x = x.tocsr()  # not csc
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


def _gnufiletype( lpfile ):
    """ lpfile.glp .mod .mps .glp .lp -> dict( glp=lpfile )
        for glpk.LPX( **_gnufiletype( lpfile ))
    """
    f = lpfile.replace( ".gz", "" )
    if f.endswith( ".glp" ):  return dict( glp=lpfile )  # gnu lp
    if f.endswith( ".lp" ):   return dict( cpxlp=lpfile )  # cplex
    if f.endswith( ".mod" ):  return dict( gmp=lpfile )  # gnu MathProg, read
    if f.endswith( ".mps" ):  return dict( freemps=lpfile )
    assert 0, ("lpfile \"%s\" should be .mod .mps .glp or .lp [optional .gz]" % lpfile)


#...............................................................................
if __name__ == "__main__":
    import sys
    from glp.zutil import scan_args, globs

    np.set_printoptions( threshold=20, edgeitems=10, linewidth=140,
            formatter = dict( float = lambda x: "%.2g" % x ))  # float arrays %.2g
    print( "\n" + 80 * "=" )

    lpfiles = "../data/glp/01/04-plan.glp"  # plan.glp: i 8 d 250 300
    # lpfiles = "../netlib/zib/mps/*.mps.gz"
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
            gnulp = save_lp( "tmp.mod", lp )  # .mod: write_modfile
