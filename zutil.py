#!/usr/bin/env python
""" zutil.py: Bag ... """

from __future__ import division, print_function
import glob
from six import string_types
import sys
import time
import numpy as np
from scipy import sparse

class Bag( dict ):
    """ a dict with d.key short for d["key"], aka Dotdict """
    def __init__(self, *args, **kwargs):
        dict.__init__( self, *args, **kwargs )
        self.__dict__ = self


def boundsvec( x, n, none ):
    """ -> n-vec, None -> none e.g. 0 """
    if np.isscalar( x ):
        return x * np.ones( n )
    x = np.squeeze( x )
    assert x.shape == (n,), [x.shape, n]
    x = np.asarray( x, dtype=float )  # None -> NaN
    x[ np.isnan(x) ] = none
    return x


def dot0( A, dot=".", mul=1 ):
    """ make big sparse arrays more readable: A*mul | ints | 0 -> dot """
    import re
    Adense = getattr( A, "A", A ) * mul
    s = str( Adense.astype(int) )  # NaN Inf -> - maxint
    return re.sub( r"\b 0 \b", dot, s, flags=re.X )


def inftonone( x ):
    """ x inf -> None for linprog lb ub """
    jinf = ~ np.isfinite( x.astype( float ))
    x = x.astype( object )
    x[jinf] = None
    return x


def ints( X ):
    return np.round(X).astype(int)  # NaN Inf -> - maxint


def maxnorm( X ):
    X = getattr( X, "data", X )  # sparse
    return np.linalg.norm( X.reshape(-1), ord=np.inf )


def minavmax( x, fmt="%.3g " ):
    return (3 * fmt) % (x.min(), x.mean(), x.max())


def nbytes( A ):
    return np.nbytes( A ) if not sparse.issparse(A) \
        else A.data.nbytes + A.indices.nbytes + A.indptr.nbytes 


def ptime( msg=None, T=[0,0]):
    """ print delta wall clock time, cpu time (sum all cores ?) from previous call
        ptime()  # no print
        ...
        ptime( "message" )
    """
    wall = time.time()  # wallclock
    cpu = getattr( time, "process_time", time.clock )()  # py3 py2
    dwall = wall - T[0]
    dcpu = cpu - T[1]
    if msg:
        print( "time: %4.1f %4.1f sec  %s" % (
                dwall, dcpu, msg ))
    T[0] = wall
    T[1] = cpu
    return dwall, dcpu


def quantiles( x, q = [0, 10, 25, 50, 75, 90, 100] ):
    return "quantiles: %s" % np.percentile( x, q=q )


def scan_args( sysargv, help="sorry, no help" ):
    """ run my.py [a=1 b=None 'c = expr' ...] [file* ...] in shell or IPython
        my.py:
            # params --
            a = 0
            eqargs, fileargs = scan_args( sys.argv )
                # expands file*
            for arg in eqargs:
                exec( arg )  # -> globals
            # print params
            for filename in fileargs or ...:
                ...
    """
    if sysargv[-1] in ("-h", "--help"):
        print( help )
        sys.exit( 0 )
    eqargs = []
    jeq = 1
    for arg in sysargv[1:]:
        if "=" in arg:
            eqargs.append( arg )
            # exec( arg, locals_ )  # not py3 -- readonly
            jeq += 1
        else:
            break
    argfiles = sysargv[jeq:]  # may be [] [""] ["-x" ""]
    return eqargs, globs( argfiles )


def globs( listoffiles ):
    """ expand file* ... like shell, ~user $var too """
    from os.path import expanduser, expandvars

    if isinstance( listoffiles, string_types ):
        listoffiles = [listoffiles]
    gfiles = []
    for file in filter( len, listoffiles ):
        gfile = glob.glob( expanduser( expandvars( file )))
        if not gfile:
            raise IOError( "file \"%s\" not found" % file )
        gfiles.extend( gfile )
    return gfiles


    # from $etc/sparseutil.py --
def sparsematrix_dict( A, nm="A" ):
    """ np.savez( mynpz, ... **sparsematrix_dict( A )) """
    if type(A).__name__ not in "csr_matrix csc_matrix ".split():
        A = A.tocsr()
    return {
        nm + "_data"    : A.data,
        nm + "_indices" : A.indices,
        nm + "_indptr"  : A.indptr,
        nm + "_shape"   : np.array( A.shape ),  # array for np.savez
        nm + "_dtype"   : type(A).__name__
        }

def dict_sparsematrix( adict, nm="A" ):
    data    = adict[ nm + "_data" ]
    indices = adict[ nm + "_indices" ].astype( int )
    indptr  = adict[ nm + "_indptr" ].astype( int )
    shape   = tuple( adict[ nm + "_shape" ])
    dtype   = str( adict[ nm + "_dtype" ])  # csr_matrix csc_matrix
    mat = getattr( sparse, dtype )
    return mat( (data, indices, indptr), shape=shape )


#...........................................................................
if __name__ == "__main__":
    print( "\n" + 80 * "=" )
    print( " ".join(sys.argv) )
    x = 0
    eqargs, fileargs = scan_args( sys.argv )
    for arg in eqargs:
        exec( arg )
    print( "x: %s  eqargs: %s  fileargs: %s " % (x, eqargs, fileargs) )

