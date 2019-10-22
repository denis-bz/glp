#!/usr/bin/env python
""" load_lp( lpfile, to="glpk | lp | linprog" ) """

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

        GLPK file formats: see http://lpsolve.sourceforge.net/5.5/formulate.htm
    """
    open( lpfile )  # else IOError: No such file or directory
    assert to in "glpk lp linprog glpk-glpk ".split(), to

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

    if to == "glpk-glpk":  # loopback test glpk - lp - linprog - lp - glpk
        lp = glp.linprog_to_lp( linrec, name=gnulp.name ) 
        return glp.lp_to_gnulp( lp, verbose=verbose )
    assert 0, to


def save_lp( outfile, lp ):
    """ LP() or gnulp -> outfile .mod .mps ... """
    if isinstance( lp, Bag ):  # lprec
        lp = glp.lp_to_gnulp( lp )
    # assert isinstance( lp, glpk.LPX )
    lp.write( **_gnufiletype( outfile ))  # .mod .mps ...
    return lp
    

def _gnufiletype( lpfile ):
    """ lpfile.glp .mod .mps .glp .lp -> dict( glp=lpfile )
        for glpk.LPX( **_gnufiletype( lpfile ))
    """
    f = lpfile.replace( ".gz", "" )
    if f.endswith( ".glp" ):  return dict( glp=lpfile )  # gnu lp
    if f.endswith( ".lp" ):   return dict( cpxlp=lpfile )  # cplex
    if f.endswith( ".mod" ):  return dict( gmp=lpfile )  # gnu MathProg
    if f.endswith( ".mps" ):  return dict( freemps=lpfile )
    assert 0, ("lpfile \"%s\" should be .mod .mps .glp or .lp [optional .gz]" % lpfile)


#...............................................................................
if __name__ == "__main__":
    import sys
    from glp.zutil import scan_args, globs

    np.set_printoptions( threshold=20, edgeitems=10, linewidth=140,
            formatter = dict( float = lambda x: "%.2g" % x ))  # float arrays %.2g
    print( "\n" + 80 * "=" )

    lpfiles = "../data/glp/01/04-plan.glp"  # ../bin/glp-sizes
    # lpfiles = "../netlib/zib/mps/*.mps.gz"
    # lpfiles = "../netlib/zib/mps/brandy.mps.gz"  # A 219, b 220 ??  ship* too
    nin = 10
    save = 0
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
            gnulp = save_lp( "tmp.glp", lp )  # lp_to_gnulp .write
