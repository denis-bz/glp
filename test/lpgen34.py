#!/usr/bin/env python
""" lpgen34.py LP testcase generator
    d=3: 3n^2 x n^3, 3 1s in each column
        n=55: A 9075 x 166375, 499125 nnz, glpsol simplex 100 minutes
    d=4: 4n^3 x n^4, 4 1s in each column, n in each row
        n=16: A 2^14 x 2^16, 2^18 nnz, glpsol simplex 10 hours
"""
# keywords: linear programming, test case, generator, Latin-square
# https://stackoverflow.com/questions/57936789/many-vertex-test-problems-for-the-simplex-method
# https://math.stackexchange.com/questions/3370934/3d-permutation-matrices  -- 4d too

from __future__ import division, print_function  
import numpy as np
from scipy import sparse as sp

__version__ = "2019.10.08"  # 8 oct
__author_email__ = "denis-bz-py t-online.de"

#...............................................................................
def lpgen34( n, d=3, cint=9, seed=0, verbose=2 ):
    """ -> A b c LP test: d=3: A 3n^2 x n^3, 3 1s in each column  sparse csr
        b = 1
        c randint 0 .. cint / uniform
    """
        # csr ? csc most efficient for sksparse.cholmod
    Agen = { 2: A2, 3: A3, 4: A4 }[d]
    A = Agen( n )
    nr, nc = A.shape
    b = np.ones( nr )
    random = seed if isinstance(seed, np.random.RandomState) \
            else np.random.RandomState( seed=seed )
    if cint > 0:
        c = random.randint( 0, cint+1, size=nc ).astype(float)  # 0 .. cint inclusive
    else:
        c = random.uniform( size=nc )
    if verbose:
        print( "\nlpgen34: n %d  d %d  A %s %s %d non0  seed %d  c %s ..." % (
                n, d, A.shape, type(A).__name__, A.nnz, seed, c[:10] ))
        if verbose >= 2:
            for j in range( 0, nr, nr // d ):
                print( A[j:j+5] .A )  # caller's printoptions
                print( " ..." )
    return A, b, c


def A2( n ):
    """ -> A 2n x n^2, 2 1s in each column  sparse csr """
        # ~ lpgen_2d 2014
    square = np.arange( n**2 ).reshape( n, n )
    line = np.arange( n )

    def row01( *ix ):
        row = sp.lil_matrix( (1, n**2) )
        row[0, square[ix]] = 1
        return row.tocsr()

    return sp.vstack(  # csr* -> csr, else coo
            #   i  j
        [row01( i, line ) for i in line] +
        [row01( line, j ) for j in line]
        )

def A3( n ):
    """ -> A 3n^2 x n^3, 3 1s in each column  sparse csr """
    cube = np.arange( n**3 ).reshape( n, n, n )
    line = np.arange( n )
    pairs = np.array([ [i, j] for i in line  for j in line ])

    def row01( *ix ):
        row = sp.lil_matrix( (1, n**3) )
        row[0, cube[ix]] = 1
        return row.tocsr()

    return sp.vstack(
            #   i  j  k
        [row01( i, j, line ) for i, j in pairs] +
        [row01( i, line, k ) for i, k in pairs] +
        [row01( line, j, k ) for j, k in pairs]
        )

def A4( n ):
    """ -> A 4n^3 x n^4, 4 1s in each column  sparse csr """
    cube = np.arange( n**4 ).reshape( n, n, n, n )
    line = np.arange( n )
    triples = np.array([ [i, j, k] for i in line  for j in line  for k in line ])

    def row01( *ix ):
        row = sp.lil_matrix( (1, n**4) )
        row[0, cube[ix]] = 1
        return row.tocsr()

    return sp.vstack(
            #   i  j  k, l
        [row01( i, j, k, line ) for i, j, k in triples] +
        [row01( i, j, line, l ) for i, j, l in triples] +
        [row01( i, line, k, l ) for i, k, l in triples] +
        [row01( line, j, k, l ) for j, k, l in triples]
        )

#...............................................................................
if __name__ == "__main__":
    import sys

    np.set_printoptions( threshold=20, edgeitems=32, linewidth=150,
            formatter = dict( float = lambda x: "%.2g" % x ))  # float arrays %.2g

    d = 4
    n = 16
    cint = 9  # c randint 0 .. cint+1 / uniform
    bmul = 1
    save = ""  # > save.lp
    seed = 0

    # to change these params, run this.py  a=1  b=None  c='expr' ...
    # in sh or ipython --
    for arg in sys.argv[1:]:
        exec( arg )

    A, b, c = lpgen34( n, d=d, cint=cint, seed=seed )

    if save:
        import glp  # numpy/scipy arrays <-> glpk

        b *= bmul
        lp = glp.LP( A=A, b=b, blo=b, c=c, problemname=save )  # Ax = b, 0 <= x
        glp.save_lp( save + ".lp.gz", lp )  # .glp / .lp cplex / .mod gmpl / .mps

