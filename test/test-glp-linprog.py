#!/usr/bin/env python3
""" test-glp-linprog.py: lp files -> gnulp_to_lp -> lp_to_linprog -> linprog """

from __future__ import division, print_function
import sys
import time
import numpy as np
import scipy
from scipy.optimize import linprog  # $scopt131/_linprog.py $scopt131/_linprog_ip.py
    # https://docs.scipy.org/doc/scipy/reference/optimize.linprog-interior-point.html
    # + print _presolve _get_Abc in out
try:
    import sksparse  # $lopy/scikit-sparse-0.4.4/doc/cholmod.rst
except ImportError:
    sksparse = None

import glp
from glp.zutil import scan_args, globs, ptime, sparsematrix_dict

assert sys.version_info.major == 3  # scipy 1.3: py3 only

np.set_printoptions( threshold=10, edgeitems=5, linewidth=140,
        formatter = dict( float = lambda x: "%.2g" % x ))  # float arrays %.2g

print( "\n" + 80 * "=" )
print( "from", " ".join( sys.argv ), " ", time.strftime( "%c" ))
print( "versions: numpy %s  scipy %s  sksparse %s  python %s " % (
        np.__version__, scipy.__version__,
        getattr( sksparse, "__version__", None ),
        sys.version.split()[0] ))

#...............................................................................
lpfiles = "data/glp/01/*.glp"  # from glpk/examples .mod -> glp
# lpfiles = "netlib/zib/mps/*.mps.gz"  # ../bin/netlib-sizes  not brandy ship*
nin = 20
method = "interior-point"
cholesky = True  # default True
maxiter = 100
presolve = True
tol = 1e-5  # means ? default 1e-8
save = 0  # > tag.npz
tag = "tmp"

    # run my.py [a=1 b=None 'c = expr' ...] [file* ...] in shell or IPython
eqargs, fileargs = scan_args( sys.argv )
for eqarg in eqargs:
    exec( eqarg )

options = dict( cholesky=cholesky, disp=True, maxiter=maxiter, presolve=presolve,
            sparse=True, tol=tol )
print( "linprog options:" )
for k, v in sorted( options.items() ):
    print( "%-8s : %s" % (k, v) )

for lpfile in fileargs \
        or globs( lpfiles )[:nin]:
    print( "\n" + 80 * "-" )
    print( "--", lpfile )

    lp, linrec = glp.load_lp( lpfile, to="linprog" )
        # lp: Bag( A b c blo lb ub ) after rows_le
        # linrec: Bag( A_ub b_ub A_eq b_eq bounds c ), [A_ub A_eq] x <= [b_ub b_eq]
    ptime()

    #...........................................................................
    res = linprog( method=method, options=options, **linrec )
    ptime( "linprog %s" % lp.problemname )
    f, x, niter = res.fun, res.x, res.nit
        # if not disp:
        #     print( "linprog:", res.message )
    print( "final f: %g  %s  %s  niter: %d " % (
            f, lpfile, method, niter ))
    print( "x:", x )  # print_options
    print( "c:", lp.c )
    glp.lp_check( lp, x )  # lp after rows_le

    if save:
        out = tag + ".npz"
        print( "\nsaving A %s ...  > %s" % (lp.A.shape, out) )
        A, b, c, blo, lb, ub, problemname \
            = lp.A, lp.b, lp.c, lp.blo, lp.lb, lp.ub, lp.problemname
        np.savez( out,
            f=f, x=x, b=b, c=c, blo=blo, lb=lb, ub=ub, problemname=problemname,
            **sparsematrix_dict( A ))

