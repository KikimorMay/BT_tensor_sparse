import numpy as np
from numpy.linalg import pinv
from pre import fold, unfold, multi_mode_dot, check_random_state
import scipy.linalg
import scipy
import scipy.io as io
import numpy as np
import scipy.linalg
from numpy.linalg import pinv
import scipy.sparse.linalg
from scipy.misc import face, imresize
import time

import sys
#print('the line is:', sys._getframe().f_lineno, )


a = np.random.randn(3,4)
b = np.random.randn(4,5)
print(a.dot(b))



print(b)







