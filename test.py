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
from ADMM_sparse import *

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


# rank = [20,20,20]
# data = np.random.randn(10,10,10)
# data__test = constract_data(20, [100, 100, 100], 0.1)
# e = 0
#
# for i in range(20):
#     a = np.random.randn(100, 1)
#     b = np.random.randn(100, 1)
#     d = np.random.randn(100, 1)
#     c = np.outer(a, b)
#     e += np.outer(c, d).reshape(100, 100, 100)
#
# input_data = e + 0.2*np.random.randn(100,100,100)
#
# core, factors = partial_tucker(input_data, modes = [1,2,3], ranks = rank)
# recons = multi_mode_dot(core, factors,modes= [1,2,3], transpose=False)
# print('与原tensor的误差是：',norm(recons - e, 2)/norm(e, 2))
#
# core_ad, factors_2 = ADMM(data__test,ranks=[30, 30, 30], Lambda=1000)

"""
lasso_test_complex
"""
#
# def prox_Lasso(tensor, para):
#     e = tensor
#     shape = tensor.shape
#     data_plus = np.zeros(shape)
#     data_neg = np.zeros(shape)
#     a = tensor > para
#     b = tensor < -para
#     c = a|b
#     data_plus[a] = -para
#     data_neg[b] = para
#     e[c == False] = 0
#     return e + data_neg + data_plus
#
#
# def pro_Lasso_2(tensor, para):
#     shape = tensor.shape
#     for i in range(shape[0]):
#
#         if tensor[i] < -para:
#             tensor[i] = tensor[i]+ para
#         elif (tensor[i] > para):
#             tensor[i] = tensor[i] - para
#         else:
#             tensor[i] = 0
#     return tensor
# b = np.random.randn(400000000)
#
# time_1 = time.time()
# c = prox_Lasso(b, 0.5)
# time_2 = time.time()
# minus = time_2 - time_1
#
# time_3 = time.time()
# d = pro_Lasso_2(b, 0.5)
# time_4 = time.time()
# minus_2 = time_4 - time_3
# err = norm(c-d, 2)
#
# print(minus, minus_2, err)

from PIL import Image
# img=np.array(imresize(Image.open('Dataset/street/1.jpg'), 0.1),dtype='float64'  )

img=Image.open('Dataset/street/4.jpg').convert('L')  #打开图像
box=(100,100,1200,500)
roi=np.array(imresize(img.crop(box), 0.2), dtype='float64')

ranks = [10, 10]
ans_except_sparse, sparse_tensor = admm_sparse(roi, 10, tau_sparse=0.01, Rho=1, modes=[1, 2], n_part=2, ranks=ranks)
_, _, ans_tucker = partial_tucker(roi, modes=[1,2], ranks=ranks)

fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
ax.set_axis_off()
ax.imshow(to_image(roi))
ax.set_title('original')

ax = fig.add_subplot(1, 3, 2)
ax.set_axis_off()
ax.imshow(to_image(ans_except_sparse))
ax.set_title('ans_except_sparse')


ax = fig.add_subplot(1, 3, 3)
ax.set_axis_off()
ax.imshow(to_image(ans_tucker))
ax.set_title('tucker')


plt.tight_layout()
plt.show()









