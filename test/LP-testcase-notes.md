#### Some LP testcases

Netlib: http://www.zib.de/koch/perplex/data/netlib/mps
are ancient standard testcases.
See the runtimes in `glpk-netlib.md` under https://gist.github.com/denis-bz:
dfl001 30 seconds, pilot87 7 sec, the rest in <= 5 seconds.
(Moore's laws of CPU and memory are making some old test cases irrelevant,
and others suffer from versionitis.)

`Klee_Minty_cube.py` generates D x D matrices / polytopes with 2^D vertices.
Seems these were once difficult for simplex solvers;
`glpsol` does D=200 in < 0.05 seconds;
`linprog` revised-simplex in scipy 1.3 takes 70 sec for D=20
and quits before reaching the correct minimum.
The Klee-Minty `A` is very ill-conditioned;
this makes plain linear `Ax = b` difficult, with x >= 0 too ?

`lpgen34.py` generates biggish sparse problems, for example
d=4 n=16 -> A 2^14 x 2^16, 2^18 nnz.
`glpsol` simplex runs this for 10 hours
(single core, `glpsol` doesn't do multi-core).
Runtimes of other simplex solvers might be ... boring.

