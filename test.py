import numpy as np
from numpy.linalg import pinv
from pre import *
import scipy.linalg
import scipy
import scipy.io as io
import numpy as np
import scipy.linalg
from numpy.linalg import pinv
import scipy.sparse.linalg
from scipy.misc import face, imresize
import time
from ADMM import *
from admm_hosvd import *

import sys
#print('the line is:', sys._getframe().f_lineno, )


# a = np.random.randn(30, 20)
# U, S, V= partial_svd(a,25)
# k = np.zeros([U.shape[1], V.shape[0]])
# for index, num in enumerate(S):
#     k[index, index] = num
# print(U.shape, k.shape, V.shape)
#
# print(norm(a-U.dot(k).dot(V), 2 ))


rank = [20,20,20]
data = np.random.randn(10,10,10)
data__test = constract_data(20, [100, 100, 100], 0.1)
e = 0

for i in range(20):
    a = np.random.randn(100, 1)
    b = np.random.randn(100, 1)
    d = np.random.randn(100, 1)
    c = np.outer(a, b)
    e += np.outer(c, d).reshape(100, 100, 100)

e = e + 0.2*np.random.randn(100,100,100)

partial_tucker(e, modes = [1,2,3], ranks = rank)
ADMM(data__test,ranks=[30, 30, 30], Lambda=1000)






