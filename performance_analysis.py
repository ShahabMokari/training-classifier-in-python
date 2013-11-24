import pstats

p = pstats.Stats('nb_clf_log.pyprof')
p.sort_stats('cumulative').print_stats()
