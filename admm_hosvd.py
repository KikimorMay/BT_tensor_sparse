import numpy as np
from pre import *


def ADMM_hosvd(tensor, Lambda, mu, iterations= 100, tol = 1e-5):   #自动选取秩！
    shape_tensor = tensor.shape
    mul = 1
    for i in shape_tensor:
        mul *= i
    n_mode = len(shape_tensor)

    tao = 3*mu

    #initial:
    M = []
    Y = []
    for index in range(n_mode):
        append_tensor = np.zeros(shape_tensor)
        M.append(append_tensor); Y.append(append_tensor)
    X = np.zeros(shape_tensor)

    u_list = []
    for iteration in range(iterations):
        for index in range(n_mode):
            mode = index + 1

            #update M:
            prox_W = (mu*unfold(X, mode) - unfold(Y[index], mode))/mu

            #这里写个函数 prox
            U_prox, S_prox, V_prox = partial_svd(prox_W)
            cut_place = shape_tensor[index]
            for i, singular in enumerate(S_prox):
                if singular > 1 / mu:
                    cut_place = i + 1

            U, S, V = partial_svd(prox_W, cut_place)

            u_list.append(U)

            S_mat = np.zeros([U.shape[1], V.shape[0]])
            for i, num in enumerate(S):
                S_mat[index, index] = num
            print(S_mat.shape)
            M[index] = fold(np.dot(U, S_mat).dot(V), mode, shape=shape_tensor)

        #updata X
        mid_sum = np.zeros(shape_tensor)
        for index in range(n_mode):
            mid_sum += mu*M[index] + Y[index]
        X = ( mid_sum + Lambda*tensor)/(n_mode*mu + Lambda)

        #update Y
        for index in range(n_mode):
            Y[index] += mu*(M[index] - X)

        err_list = []

        core = multi_mode_dot(X, u_list, modes = [1,2,3], transpose=True)
        recon_tensor = multi_mode_dot(core, u_list, modes = [1,2,3], transpose=False)
        err = norm(recon_tensor-tensor, 2)/norm(tensor, 2)
        err_list.append(err)

        print('iteration is', iteration, 'err is', err)
        if iteration == 10:
            break

    return core, u_list


if __name__ =='__main__':
    e = 0

    for i in range(20):
        a = np.random.randn(100, 1)
        b = np.random.randn(100, 1)
        d = np.random.randn(100, 1)
        c = np.outer(a, b)
        e += np.outer(c, d).reshape(100, 100, 100)

    data  = e + 0.2 * np.random.randn(100, 100, 100)
    rank = [20, 20, 20]
    core_tucker, factor_tucker = partial_tucker(data, modes=[1, 2, 3], ranks=rank)
    recons_tucker = multi_mode_dot(core_tucker, factor_tucker, modes = [1,2,3], transpose=False)
    ans = norm(recons_tucker-e, 2)/norm(data, 2)
    print(ans)

    core, factor_list = ADMM_hosvd(data, 100, 0.02)
    recon_tensor = multi_mode_dot(core, factor_list, modes=[1, 2, 3], transpose=False)
    print(norm(recon_tensor-e, 2)/norm(data, 2))



