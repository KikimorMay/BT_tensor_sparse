from pre import *
import numpy as np


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


    for iteration in range(iterations):
        for index in range(n_mode):
            mode = index + 1

            #update M:
            prox_W = (mu*unfold(X, mode) - unfold(Y[index], mode) + tao*unfold(M[index], mode))/(mu + tao)
            U_prox, S_prox, V_prox = partial_svd(prox_W)
            cut_place = shape_tensor[index]
            for i, singular in enumerate(S_prox):
                if singular > 1 / (mu + tao):
                    cut_place = i + 1

            U, S, V = partial_svd(prox_W, cut_place)

            S_mat = np.zeros([U.shape[1], V.shape[0]])
            for i, num in enumerate(S):
                S_mat[index, index] = num
            print(S_mat.shape)
            M[index] = fold(np.dot(U, S_mat).dot(V), mode, shape=shape_tensor)

        #updata X
        mid_sum = np.zeros(shape_tensor)
        for index in range(n_mode):
            mid_sum += mu*M[index] + Y[index]
        X = ( mid_sum + Lambda*tensor + tao*X)/(n_mode + Lambda + tao)

        #update Y
        for index in range(n_mode):
            Y[index] += mu*(M[index] - X[index])

        err = norm(X-tensor,2)/norm(tensor, 2)
        print('iteration is', iteration, 'err is', err)
        if err < tol:
            break

if __name__ =='__main__':
    e = 0

    for i in range(20):
        a = np.random.randn(100, 1)
        b = np.random.randn(100, 1)
        d = np.random.randn(100, 1)
        c = np.outer(a, b)
        e += np.outer(c, d).reshape(100, 100, 100)

    e = e + 0.02 * np.random.randn(100, 100, 100)

    ADMM_hosvd(e, 100, 0.02)



