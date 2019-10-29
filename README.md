#### glp: numpy <--> GLPK, the gnu linear programming kit

Keywords, tags: linear programming, python, numpy, scipy, GLPK, GMPL, bridge

`glp` is a lightweight bridge between
the Gnu Linear Programming Kit [GLPK](https://www.gnu.org/software/glpk/)
and the numpy-scipy-python world:

	numpy/scipy arrays  <-->  LP( A b c ... )  <-->  LP files

* numpy and scipy help construct large sparse LP models, and connect to dozens of specialized tools
* GLPK reads and writes LP models in several formats: .mod, .lp (aka cplex), .mps

Requirements: [GLPK](https://www.gnu.org/software/glpk/) and
[PyGLPK](https://github.com/bradfordboyle/pyglpk) -- see "installing" below.


#### Example: file -> numpy --

    import glp
    lp = glp.load_lp( ".../my.mps" )  # an LP record with lp.A lp.b lp.c ..., see below


#### Example: numpy -> file -> solver --

    # make numpy/scipy arrays A b c ... that describe your problem
    # and put them in a Bag, see "LP standard form" and "LP()" below --
		lp = glp.LP( A= b= c= blo= lb= ub= )

    # save A b c ... in a text file --
		gnulp = glp.save_lp( "my.lp", lp )  # or my.mps or my.glp

    # run my.lp -> your solver in a shell or a python subprocess
    # read, plot the solution file


#### LP standard form: `A b c blo lb ub`

In `glp`, LP problems are described by a record (struct, Bag)
with 6 numpy arrays `A, b, c, blo, lb, ub`:

    minimize c . x
    subject to
        blo <= Ax <= b   # row bounds, default Ax <= b
        lb  <= x  <= ub  # column (variable) bounds, default 0 <= x

In this form, the row bounds `b blo` and column (variable) bounds `lb ub`
are nicely symmetric,
and cover all combinations of 1-sided and 2-sided bounds. For `lb <= x <= ub`:

    Lower only: lb   .. inf  -- no upper bound
    Upper only: -inf .. ub   -- no lower bound
    Both:       lb   .. ub
    Equal:      lb   == ub
    no bounds:  -inf .. inf

Row bounds `blo <= Ax <= b` have the same 5 cases.
(Rows with no bounds are irrelevant, so `glpsol` removes them, `lp_to_linprog` too.)


#### The main data structure: `LP( A, b, c ... )`

    lp = glp.LP( A, b, c, blo=-np.inf,  lb=0, ub=np.inf,  problemname="name" )

puts `A b c ...` into a `Bag` with fields `lp.A lp.b ...`.
`A` is made [sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html)
if it isn't already.
Bounds arrays `b blo lb ub` are expanded to numpy arrays of float64
of the appropriate size.
These may have `+-inf`, but no `None`, no `NaN`.  
`LP( ... blo=-np.inf )`, the default, constrains all rows `Ax <= b`;  
`LP( ... blo=b )` constrains `Ax = b`.

(A Bag aka Dotdict aka flexible struct
is a dict with `bag.key` short for `bag["key"]` --
easier to pass around than long arg lists,
and `bag.<tab>` in IPython shows you the fields.
See the 5-line `class Bag( dict )` in `zutil.py`.)


#### File and function overview

    lprec.py
        def LP( A, b, c, blo=-inf, lb=0, ub=inf,
        def print_lp( lp, verbose=1, header="", footer="" ):

    load_lp.py
        def load_lp( lpfile, to="lp", verbose=1 ):
                """ GLPK file .mod .mps ...  to= "lp" | "linprog" | "glpk"
        def save_lp( outfile, lp ):
                """ LP() or gnulp -> outfile .mod .mps ... """

    lp_gnu.py
        def gnulp_to_lp( gnulp, drop_uncon=True, verbose=1 ):
                    """ gnulp .matrix .rows .cols .obj -> LP( A b c ... ) numpy/scipy arrays
        def lp_to_gnulp( lp, verbose=1 ):
                    """ LP( A b c blo lb ub ... ) -> glpk .matrix .rows .cols ...
        # def gnulp_solve( gnulp, solver="interior", verbose=1 ):
        #             better save_lp | glpsol | load_lp

    lp_linprog.py
        def lp_to_linprog( lp, verbose=1 ):
                    """ LP( A b c ... ), see lprec.py
        def linprog_to_lp( linp, name="noname" ):
                    """ test-loopback.py: lp_to_linprog, linprog_to_lp
        def rows_le( A, b, blo ):
                    """ -inf <= Ax <= inf: drop these rows, unconstrained
        def split_Ab( A, b, blo ):
                    """ -> Bag( A_ub b_ub A_eq b_eq )

    zutil.py
        class Bag( dict ):
            """ a dict with d.key short for d["key"], aka Dotdict """
        def scan_args( sysargv, globals_, locals_ ):
            """ run my.py [a=1 b=None 'c = expr' ...] [file* ...] in shell or IPython


#### Notes on GLPK and `glpsol`

GLPK reads and writes LP problems in various formats:

* .lp aka CPLEX format
* .mps, an old format used for e.g. Matlab <-> solvers
* .glp, easy to parse in python or awk
* .mod, the gnu MathProg language [GMPL](https://en.wikibooks.org/wiki/GLPK/GMPL_(MathProg)).
This is roughly a subset of the [AMPL](https://ampl.com) language.

`glp.save_lp( "my.mod", lp )` writes a `.mod` file similar to `.lp` aka cplex
which is readable in AMPL -- in simple cases, mttiw.

GLPK's solver `glpsol` has over 50 options, for simplex, interior-point, and mixed-integer (MIP).
It carries the user's constraint names and variable names through to solution files;
this is essential for making solutions to big problems understandable --
splitting, sorting, plotting thousands of variables.
The simplex solver (not ip ?) does preprocessing to make problems smaller,
in some cases a good deal smaller.

To check a file, `glpsol --check --format my.file`.  
To translate e.g. `.mps` to `.lp`,  `glpsol --check --freemps in.mps --wlp out.lp`.
(`in.mps.gz` or `out.lp.gz` compress / uncompress with `gzip` on the fly.)

`glp` has a minimal `glp_solve()`, but I prefer to run `glpsol` with files, like this:

    ipython: ... save_lp( "my.glp", LP( A, b, c ... ))
    glpsol --options ...  my.glp  -w my.sol  --log my.glpsollog  # or whatever solver you like
        # see glpsol -h and bin/glpsols
    ipython: ... parse my.glp and my.sol, plot

#### Look at LP in the whole flow, input - optimize - output

For real-world optimization problems,
an LP solver (optimizer) is only part of a flow, a cycle, a process:

* input: map a problem to a sea of numbers `A b c ...`
* run the LP solver -> a sea of numbers `x[...]`
* map back: make the solution `x[...]` *understandable*, with plots and talks
* <sub> (sotto voce) check that there are no mistakes along the way. </sub>

Experts say that commercial solvers are much faster than GLPK,
but GLPK may be fast enough for *your* problem.


#### Installing glpk and pyglpk

    # first glpk:
	download and unpack https://www.gnu.org/software/glpk/.../glpk-4.65.tar.gz
	configure --prefix=/opt/local;  make;  make install

    # pyglpk:
    pip install --user git+https://github.com/bradfordboyle/pyglpk
		# (git clone gets examples/*.py too)

    # test:
    ipython
        import glpk          # i.e. pyglpk
        print glpk.__file__  # .../glpk.so
        gnulp = glpk.LPX()   # empty
        gnulp = glpk.LPX( gmp="xx.mod" )  # Reading ... Generating ...


#### Links

[GLPK](https://www.gnu.org/software/glpk/)  
[PyGLPK on github](https://github.com/bradfordboyle/pyglpk)  
[PyGLPK brief reference](https://github.com/bradfordboyle/pyglpk/blob/master/docs/brief-reference.rst)  
[scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html)  
[LPsolve FAQ](http://lpsolve.sourceforge.net/5.0/LinearProgrammingFAQ.htm)  
[LPsolve doc on file formats](http://lpsolve.sourceforge.net/5.5/formulate.htm)  
[Numerical Recipes p. 537-549: Linear Programming Interior-Point Methods](http://apps.nrbook.com/empanel/index.html?pg=537)  
GLPK runtimes for Netlib and other test cases are under [my gists](https://gist.github.com/denis-bz)  


#### Comments welcome, test cases welcome.

cheers  
  -- denis 29 October 2019

