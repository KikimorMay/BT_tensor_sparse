

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

def t_SVD(tensor, svd_eig=None):
    """
    :param tensor:
    :param svd_eig: 一定要小于shape[0]和shape[1]
    :return:
    """
    tensor_shape = tensor.shape
    I, J, K = tensor.shape

    U_tensor = np.zeros([I, svd_eig, K])
    V_tensor = np.zeros([J, svd_eig, K])
    S_tensor = np.zeros([svd_eig, svd_eig, K])
    ans_tensor = np.zeros(tensor_shape)
    for k in range(tensor_shape[-1]):
        U, S, V_transpose = partial_svd(tensor[:, :, k], svd_eig)
        U_tensor[:, :, k] = U
        S_tensor[:, :, k] = np.diag(S)
        V_tensor[:, :, k] = V_transpose.T
    U_tensor = U_tensor
    S_tensor = S_tensor
    V_tensor = V_tensor

    middle_ans = np.zeros([I, svd_eig, K])
    for k in range(K):
        for i in range(middle_ans.shape[0]):
            for j in range(middle_ans.shape[1]):
                middle_ans[i][j][k] = U_tensor[i][j][k] * S_tensor[j][j][k]
    for k in range(K):
        for i in range(I):
            for j in range(J):
                for p in range(svd_eig):
                    ans_tensor[i][j][k] += middle_ans[i][p][k] * V_tensor[j][p][k]

    print(norm(tensor-ans_tensor, 2)/norm(tensor, 2))
    return

def t_SVD_fft(data, svd_eig=None):
    """
    ##加了fft不对啊妈也
    :param tensor:
    :param svd_eig: 一定要小于shape[0]和shape[1]
    :return:
    """
    tensor_shape = data.shape
    I, J, K = data.shape
    tensor = np.fft.fft(data)

    U_tensor = np.zeros([I, svd_eig, K])
    V_tensor = np.zeros([J, svd_eig, K])
    S_tensor = np.zeros([svd_eig, svd_eig, K])
    ans_tensor = np.zeros(tensor_shape)
    for k in range(tensor_shape[-1]):
        U, S, V_transpose = partial_svd(tensor[:, :, k], svd_eig)
        U_tensor[:, :, k] = U
        S_tensor[:, :, k] = np.diag(S)
        V_tensor[:, :, k] = V_transpose.T
    U_tensor = np.fft.ifft(U_tensor)
    S_tensor = np.fft.ifft(S_tensor)
    V_tensor = np.fft.ifft(V_tensor)

    middle_ans = np.zeros([I, svd_eig, K])
    for k in range(K):
        for i in range(middle_ans.shape[0]):
            for j in range(middle_ans.shape[1]):
                middle_ans[i][j][k] = U_tensor[i][j][k] * S_tensor[j][j][k]
    for k in range(K):
        for i in range(I):
            for j in range(J):
                for p in range(svd_eig):
                    ans_tensor[i][j][k] += middle_ans[i][p][k] * V_tensor[j][p][k]

    print(norm(data-ans_tensor, 2)/norm(data, 2))