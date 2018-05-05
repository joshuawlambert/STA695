import numpy as np
import sys

fname = sys.argv[1]
acc = np.loadtxt(fname)
print('{}+/-{}'.format(acc.mean(), acc.std()))
