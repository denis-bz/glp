#!/usr/bin/env python
""" test-glp-glpk.py: lp files -> gnulp_to_lp -> gnulp_solve """

from __future__ import division, print_function
import sys
import time
import numpy as np
import scipy

import glp
from glp.zutil import scan_args, globs, ptime

np.set_printoptions( threshold=20, edgeitems=10, linewidth=140,
        formatter = dict( float = lambda x: "%.2g" % x ))  # float arrays %.2g
print( "\n" + 80 * "=" )
print( "from", " ".join( sys.argv ), " ", time.strftime( "%c" ))
print( "versions: numpy %s  scipy %s  python %s " % (
        np.__version__, scipy.__version__, sys.version.split()[0] ))

#...............................................................................
lpfiles = "data/glp/01/01-transp.glp"  # from glpk/examples .mod -> .glp
# lpfiles = "netlib/zib/mps/*.mps.gz"  # ../bin/netlib-sizes  not brandy ship*
solver = "simplex"  # "ip"

    # run my.py [a=1 b=None 'c = expr' ...] [file* ...] in shell or IPython
eqargs, fileargs = scan_args( sys.argv )
for eqarg in eqargs:
    exec( eqarg )
np.random.seed( 0 )
print( "params: solver %s " % solver )

for lpfile in fileargs \
        or globs( lpfiles ):
    print( "\n" + 80 * "-" )
    print( "--", lpfile )

    lp = glp.load_lp( lpfile, to="lp", verbose=0 )  # Bag( A b c blo lb ub )
    print( "lp.rowname: %s ..." % " ".join( lp.rowname[:3] ))
    print( "lp.colname: %s ..." % " ".join( lp.colname[:3] ))

    gnulp = glp.lp_to_gnulp( lp )
    ptime()

    sol = glp.gnulp_solve( gnulp, solver=solver, verbose=1 )
    ptime( "glpk %s %s" % (solver, lp.problemname) )
    glp.lp_check( lp, sol.x )

