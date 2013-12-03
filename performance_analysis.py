import pstats

p = pstats.Stats('nb_clf.pyprof')
p.sort_stats('cumulative').print_stats(20)
