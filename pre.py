#-*- coding: UTF-8 -*-
import scipy
import scipy.io as io
import numpy as np
import scipy.linalg
from numpy.linalg import pinv
import scipy.sparse.linalg
from scipy.misc import face, imresize
import sys

def constract_data(rank, size, noise):
    ans = np.zeros(size)
    for i in range(rank):
        mid_ans = np.random.randn(size[0], 1)
        n = len(size) - 1
        for index in range(n):
            b = np.random.randn(size[index+1], 1)
            mid_ans = np.outer(mid_ans, b)
        mid_ans = mid_ans.reshape(size)
        ans += mid_ans

    e =  noise * rank * np.random.randn(*size)
    ans = ans + e
    return ans

def change(a, b):
    c = a
    a = b
    b = c
    return a, b

def check_random_state(seed):
    """Returns a valid RandomState
    Parameters
    ----------
    seed : None or instance of int or np.random.RandomState(), default is None
    Returns
    -------
    Valid instance np.random.RandomState
    Notes
    -----
    Inspired by the scikit-learn eponymous function
    """
    if seed is None or isinstance(seed, int):
        return np.random.RandomState(seed)

    elif isinstance(seed, np.random.RandomState):
        return seed

    raise ValueError('Seed should be None, int or np.random.RandomState')

def norm(tensor, order):     #范数
    if order == 1:
        return np.sum(np.abs(tensor))
    if order == 2:
        return np.sqrt(np.sum(tensor**2))
    else:
        return np.sum(np.abs(tensor)**order)**(1/order)

def load_mat(path):
    target_image = io.loadmat(path)['indian_pines']  # shape[145, 145]
    target_image_np = np.array(target_image, dtype=float)
    return target_image_np

def tensor_to_vec(tensor):
    return tensor.reshape(-1)

def vec_to_tensor(vec, shape):
    return vec.reshape(shape)

# 从模式一开始展开的！
def unfold(tensor, mode):
    index = mode - 1
    for i in range(tensor.ndim - index):
        for j in range(index):
            tensor = np.moveaxis(tensor, index-j+i, index-j+i-1)
    return np.reshape(tensor, (tensor.shape[0], -1))

# mode从一开始
def fold(unfold_tensor, mode, shape):
    index = mode - 1
    dim = len(shape)
    shape_new = []
    for i in range(dim):
        shape_new.append(shape[(index+i)%dim])
    tensor = unfold_tensor.reshape(shape_new)
    for i in range(dim - index):
        for j in range(index):
            tensor = np.moveaxis(tensor, dim-1-index+j-i, dim-index+j-i)

    # shape_list = list(shape)
    # mode_dim = shape_list.pop(index)
    # shape_list.insert(0, mode_dim)
    # return np.moveaxis(np.reshape(unfold_tensor, shape_list), 0, index)
    return tensor

def QR(A):
    """Gram-schmidt正交化"""
    Q = np.zeros_like(A)
    cnt = 0
    for a in A.T:
        u = np.copy(a)
        for i in range(0, cnt):
            u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i])  # 减去待求向量在以求向量上的投影
        e = u / np.linalg.norm(u)  # 归一化
        Q[:, cnt] = e
        cnt += 1
    R = np.dot(Q.T, A)
    return (Q, R)

# 计算数组们的Khatri-Rao product of a list of matrices
# 输入应该是多个列数相同的矩阵
def kr(matrices):                      # 输入n个数组
    columes = matrices[0].shape[1]      # 所有矩阵的列数应该是相等的
    n_number = len(matrices)
    start = ord('a')
    common_dim = 'z'

    target = ''.join(chr(start + i) for i in range(n_number))     # 用指定符号去链接相应的元素
    source = ','.join(i+common_dim for i in target)               # 用 , 去连接 i+target
    operation = source + '->' + target + common_dim
    return np.einsum(operation, *matrices).reshape(-1, columes)

# n_eigenvecs是需要计算的奇异值的个数
def partial_svd(matrix, n_eigenvecs=None):
    if matrix.ndim != 2:
        raise ValueError('matrix be a matrix. matrix.ndim is {} != 2'.format(
            matrix.ndim))
    dim_1, dim_2 = matrix.shape
    if dim_1 <= dim_2:
        min_dim = dim_1
    else:
        min_dim = dim_2

    if n_eigenvecs is None or n_eigenvecs >= min_dim:
        # Default on standard SVD
        U, S, V = scipy.linalg.svd(matrix)
        U, S, V = U[:, :n_eigenvecs], S[:n_eigenvecs], V[:n_eigenvecs, :]
        return U, S, V

    else:
        # We can perform a partial SVD
        # First choose whether to use X * X.T or X.T *X
        if dim_1 < dim_2:
            S, U = scipy.sparse.linalg.eigsh(np.dot(matrix, matrix.T), k=n_eigenvecs, which='LM')
            S = np.sqrt(S)
            V = np.dot(matrix.T, U * 1 / S[None, :])
        else:
            S, V = scipy.sparse.linalg.eigsh(np.dot(matrix.T, matrix), k=n_eigenvecs, which='LM')
            S = np.sqrt(S)
            U = np.dot(matrix, V) * 1 / S[None, :]

        # WARNING: here, V is still the transpose of what it should be
        U, S, V = U[:, ::-1], S[::-1], V[:, ::-1]
        return U, S, V.T

# tensor与向量模式n相乘
# 矩阵写在前面
def mode_dot(tensor, matrix_or_vector, mode):
    index = mode - 1
    list_shape = list(tensor.shape)
    new_mode = mode
    if matrix_or_vector.ndim == 2:
        if matrix_or_vector.shape[1] != tensor.shape[index]:
            raise ValueError(
                'shapes {0} and {1} not aligned in mode-{2} multiplication: {3} (mode {2}) != {4} (dim 1 of matrix)'.format(
                    tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[1]
                ))
        list_shape[index] = matrix_or_vector.shape[0]
    elif matrix_or_vector.ndim == 1:
        if matrix_or_vector.shape[0] != tensor.shape[index]:
            raise ValueError(
                'shapes {0} and {1} not aligned for mode-{2} multiplication: {3} (mode {2}) != {4} (vector size)'.format(
                    tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[0]
                ))
        if len(list_shape) > 1:
            list_shape.pop(index)
            new_mode -= 1
        else:
            list_shape = [1]
    else:
        raise ValueError('Can only take n_mode_product with a vector or a matrix.'
                         'Provided array of dimension {} not in [1, 2].'.format(np.ndim(matrix_or_vector)))
    res = np.dot(matrix_or_vector, unfold(tensor, mode))
    return fold(res, new_mode, list_shape)

def multi_mode_dot(tensor, matrix_or_vec_list, modes = None, skip = None, transpose = None):
    #skip : index of a matrix to skip. if provided, should have a length of tensor.ndim
    if modes == None:
        modes = [i+1 for i in range(len(matrix_or_vec_list))]

    decrement = 0  # If we multiply by a vector, we diminish the dimension of the tensor
    res = tensor

    for i, (matrix_or_vec, mode) in enumerate(zip(matrix_or_vec_list, modes)):
        if (skip !=None) and (i == skip-1):
            continue
        if transpose:
            res = mode_dot(res, np.transpose(matrix_or_vec), mode - decrement)
        else:
            res = mode_dot(res, matrix_or_vec, mode - decrement)
        if matrix_or_vec.ndim == 1:
            decrement == 1
    return res

# 计算两个分块矩阵的kron积， mat_1 = [A1, A2, ..., Ar], mat_2 = [B1, B2, .., Br]  分块矩阵的个数要相同
def mat_kr(mat_1, mat_2, R):
    I, LR = mat_1.shape
    L = LR//R
    J, MR = mat_2.shape
    M = MR//R
    mat_ans = np.zeros([I*J, L*M*R])
    for i in range(R):
        mat_ans[:, i*L*M:L*M*(i+1)] = np.kron(mat_1[:, i*L:(i+1)*L], mat_2[:, i*M:(i+1)*M])
    return mat_ans

# 计算 多个分块矩阵的kron积，mode 从1开始
def multi_mat_kr(matrices, R, mode = None):
    n_matrice = len(matrices)
    if mode != None:
        ans_kron = mat_kr(matrices[mode % n_matrice], matrices[(mode + 1) % n_matrice], R=R)
        i = mode + 2
        while (i % n_matrice != mode - 1):
            ans_kron = mat_kr(ans_kron, matrices[i % n_matrice], R=R)
            i += 1
    if mode == None:
        ans_kron = mat_kr(matrices[0], matrices[1], R=R)
        index = 2
        while(index < n_matrice):
            ans_kron = mat_kr(ans_kron, matrices[index], R = R)
            index += 1
    return ans_kron

def blockdiag(tensor_list, mode, p_inverse = True):
    """

    :param tensor_list: 一个list,list里面每一个元素对应的是一个core tensor, 一共有R个tensor
    :param mode:  对于每一个core tensor unfold的方式
    :param p_inverse: 是不是unfold的tensor求伪逆之后再拼接在一起
    :return: 一个矩阵 np.array
    """
    R = len(tensor_list)
    tensor_dim = tensor_list[0].ndim
    tensor_shape = tensor_list[0].shape
    blockdiag_row = 1
    for index in range(tensor_dim):
        if index == mode - 1:
            continue
        blockdiag_row *= tensor_shape[index]
    blockdiag_col = tensor_shape[mode - 1]    #对应的mode的维度上的长度

    if p_inverse == False:
        ans_mat = np.zeros([blockdiag_row*R, blockdiag_col*R])
        for index in range(R):
            ans_mat[index*blockdiag_row:(index+1)*blockdiag_row, index*blockdiag_col:(index+1)*blockdiag_col] = unfold(tensor_list[index], mode).T
    if p_inverse == True:
        blockdiag_row, blockdiag_col = change(blockdiag_row, blockdiag_col)
        ans_mat = np.zeros([blockdiag_row*R, blockdiag_col*R])
        for index in range(R):
            ans_mat[index*blockdiag_row:(index+1)*blockdiag_row, index*blockdiag_col:(index+1)*blockdiag_col] = pinv(unfold(tensor_list[index], mode).T)
    return ans_mat



 # 部分tucker分解

def rebuilt_block_term_tensor(cores, factors, modes):
    n_part = len(cores)
    n_mode = len(factors)
    shape_row = [factors[index].shape[0] for index in range(n_mode)]
    part_col = [factors[index].shape[1]//n_part for index in range(n_mode)]
    rebuilt_tensor = np.zeros(shape_row)

    factors_cal = []
    for i in range(n_part):
        factors_cal.append([])
        for j in range(len(factors)):
            factors_cal[i].append(factors[j][:, i*part_col[j]: (i+1)*part_col[j]])

    for index in range(n_part):
        rebuilt_tensor += multi_mode_dot(cores[index], factors_cal[index] ,modes= modes, transpose=False)
    return rebuilt_tensor

def partial_tucker(tensor, modes, ranks = None, init = 'SVD',  n_iter_max =100,
                   tol = 10e-5, random_state = None, verbose = True):
    #Parameters:
    #modes: int list, 列表中是需要进行分解的模式
    #rank: core tensor 的size, 和需要分解的维度的个数相同,每个值是每个维度上需要的分解的秩
    #tol: reconstruction error is less than the tolerance
    if ranks == None:
        ranks = [tensor.shape[mode-1] for mode in modes]

    #SVD init
    factors = []
    if init == 'SVD':
        for mode in modes:
            index = mode - 1
            eigenvecs, _, _ = partial_svd(unfold(tensor, mode), n_eigenvecs=ranks[index])
            factors.append(eigenvecs)
    else:
        rng = check_random_state(random_state)
        core = np.array(rng.random_sample(ranks))
        factors = [np.array(rng.random_sample((tensor.shape[mode-1], ranks[index]))) for (index,mode) in enumerate(modes)]

    rec_errors = []
    norm_tensor = norm(tensor, 2)

    for iteration in range(n_iter_max):
        for mode in modes:
            index = mode - 1
            core_approximate = multi_mode_dot(tensor, factors, modes= modes, skip= mode, transpose=True)
            eigenvecs, _, _ = partial_svd(unfold(core_approximate, mode), n_eigenvecs = ranks[index])
            factors[index] = eigenvecs
        core = multi_mode_dot(tensor, factors, modes = modes, transpose=True)
        #对tucker分解得到的部分进行重构形成新的tensor
        middle_ans = multi_mode_dot(core, factors, modes= modes, transpose=False)
        rec_error = norm(tensor - middle_ans, 2)/norm_tensor

        print(rec_error)
        rec_errors.append(rec_error)

        if iteration > 1:
            if verbose:
                print('reconsturction error={}, variation={}.'.format(
                    rec_errors[-1], rec_errors[-2] - rec_errors[-1]))

            if (rec_errors[-2] - rec_errors[-1] < tol):
                if verbose:
                    print('converged in {} iterations.'.format(iteration))
                break
    return core, factors

def block_term_tensor_decomposition(tensor, modes, n_part, ranks = None, n_iter_max = 100, tol = 10e-5, random_state = None):
    if ranks == None:
        ranks = [tensor.shape[index-1] for index in modes]

    # 随机生成core-tensor，以及factor matrixs
    rng = check_random_state(random_state)
    core = [np.array(rng.random_sample(ranks)) for i in range(n_part)]
    core_2 = [np.array(rng.random_sample(ranks)) for i in range(n_part)]

    factors = [np.array(rng.random_sample((tensor.shape[index], ranks[index]*n_part))) for index,mode in enumerate(modes)]

    rec_errors = []
    for iteration in range(n_iter_max):
        for mode in modes:
            index = mode - 1
            factors[index] = (blockdiag(core, mode).dot(pinv(multi_mat_kr(factors, R=n_part, mode = mode))).dot(unfold(tensor, mode).T)).T
            for i in range(n_part):
                factors[index][:, i*ranks[index]:(i+1)*ranks[index]], _ = QR(factors[index][:, i*ranks[index]:(i+1)*ranks[index]])
                # print("the line is:", sys._getframe().f_lineno, "factor_tensor.shape", factors[index].shape)

        #rebuilt core tensor
        for i in range(n_part):
            factors_rebult = []
            for mode in modes:
                index = mode - 1
                factors_rebult.append(factors[index][:, i*ranks[index]:(i+1)*ranks[index]])
            core[i] = multi_mode_dot(tensor, factors_rebult, modes = modes, transpose=True)


        rebuilt_tensor = rebuilt_block_term_tensor(core, factors, modes)
        err = norm(rebuilt_tensor-tensor, 2)/norm(tensor,2)

        rec_errors.append(err)

        # print("the line is:", sys._getframe().f_lineno, "vector_core.shape",vector_core.shape)
        if iteration > 3:
            if (np.abs(rec_errors[-1] - rec_errors[-2]) < tol):  # 跳出循环的条件
                break

    return core, factors, rec_errors[-1], iteration
    '''
    以下为得到的模式的转置：
    计算过程中原矩阵模式1的转置：
    '''
    for i in modes:
        print("factor %d's shape is "%(i), factors[i-1].shape)
    k = multi_mat_kr(factors, R = 2, mode = 3)
    print(k.shape)

if __name__ == '__main__':
    image = np.array(imresize(face(), 0.1), dtype='float64') #image has shape(768,1024,3)*0.3 = (230,307,3)
    data = load_mat('Indian_pines.mat')   # data has shape(145,145,220)
    data_2 = np.random.randn(30,30,30)
    ranks = [5, 5, 5]
    partial_tucker(data_2, modes=[1,2,3], ranks=ranks, init= None)
    core, factors, err , iteration = block_term_tensor_decomposition(data_2, modes=[1,2,3], ranks = ranks, n_part = 2)
    print('block_term_tucker err is:', err, 'iteration is ', iteration)


