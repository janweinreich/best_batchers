set term pdf size 10,10
set output 'baseline.pdf'

set multiplot layout 2,2

set xlabel 'seed'
set ylabel 'number of experiments'
plot for [q=2:10] 'baseline_nexp.dat' u 1:q i 0 w lp ti 'q='.q
set ylabel 'number of iterations'
plot for [q=2:10] 'baseline_niter.dat' u 1:q i 0 w lp ti 'q='.q


set xlabel 'q'
set ylabel 'number of experiments'
plot for [seed=2:11] 'baseline_nexp.dat' u 1:seed i 1 w lp ti 'seed='.(seed-2)
set ylabel 'number of iterations'
plot for [seed=2:11] 'baseline_niter.dat' u 1:seed i 1 w lp ti 'seed='.(seed-2)
