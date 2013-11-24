import pstats

p = pstats.Stats('running_log.pyprof')
p.sort_stats('cumulative').print_stats()
