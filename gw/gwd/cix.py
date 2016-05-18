
one_band_cix="""# CIX file for ctqmc! 
# cluster_size, number of states, number of baths, maximum_matrix_size
1 4 2 1
# baths, dimension, symmetry
0    1 0    0
1    1 0    0
# cluster energies for non-equivalent baths, eps[k]
0.0
#   N   K   Sz size
  1   0  0  0.0  1    3   2    0.0    0.0
  2   1  0 -0.5  1    4   0    0.0    0.5
  3   1  0  0.5  1    0   4    0.0    0.5
  4   2  0  0.0  1    0   0    0.0    0.0
# matrix elements
  1   3   1  1 1.0
  1   2   1  1 1.0
  2   4   1  1 1.0
  2   0   0  0
  3   0   0  0
  3   4   1  1 -1.0
  4   0   0  0
  4   0   0  0
HB2
# Uc = U[m1,m2,m3,m1]-U[m1,m2,m1,m3] ; loops [m1,m2,m3]
  0.000000   0.000000
  0.000000   0.000000
  0.000000   0.000000
  0.000000   0.000000
# number of operators needed
1
# Occupancy 
  1   1  1 0.0
  2   1  1 1.0
  3   1  1 1.0
  4   1  1 2.0
# Data for HB1
1 4 2 1
# ind   N   K   Jz size
  1    1   0  0  0.0  1    3   2    0.0    0   
  2    2   1  0 -0.5  1    4   0    0.0    0   
  3    3   1  0  0.5  1    0   4    0.0    0   
  4    4   2  0  0.0  1    0   0    0.0    0   
# matrix elements
  1   3   1  1 1.0
  1   2   1  1 1.0
  2   4   1  1 1.0
  2   0   0  0
  3   0   0  0
  3   4   1  1 -1.0
  4   0   0  0
  4   0   0  0
"""
