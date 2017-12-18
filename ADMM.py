from pre import *
import numpy as np

def multikrocker(matrix_list, transpose = True, skip = None):
    ans_kron = 1
    if transpose == True:
        for index in range(len(matrix_list)):
            if skip == index + 1:
                continue
            ans_kron = np.kron(ans_kron, matrix_list[index].T)
    else:
        for index in range(len(matrix_list)):
            if skip == index + 1:
                continue
            ans_kron = np.kron(ans_kron, matrix_list[index])
    return ans_kron

def ADMM(tensor, ranks, Lambda, max_iterations = 100):

    #Initialize
    mu = 1e-4
    mu_max = 10**10
    Rho = 1.05
    Tol = 1e-5
    tao = 10000
    Gamma = 100
    Y = []; G = []; factors = []

    len_ranks = len(ranks)
    all_number = 1
    for number in ranks:
        all_number *= number
    for i in range(len_ranks):
        middle_sum = all_number // ranks[i]
        y = np.zeros([ranks[i], middle_sum])
        g = np.zeros([ranks[i], middle_sum])
        u = np.random.randn(tensor.shape[i], ranks[i])
        Y.append(y); G.append(g); factors.append(u)

    # 循环直至收敛
    core = np.zeros(ranks)
    for iteration in range(max_iterations):

        # recalculate core
        middle_re = np.zeros(ranks)
        for index in range(len_ranks):
            mode = index + 1
            middle_re += fold(G[index] - Y[index], mode, ranks)
        core = (Lambda/(Lambda + len_ranks*mu))*multi_mode_dot(tensor, factors, transpose=True) + (mu/(Lambda + len_ranks*mu))*middle_re
        print('core_shape ', core.shape)
        #对每个factors, g, y 进行更新
        for index in range(len_ranks):
            mode = index + 1
            # update factors[n]
            W = np.dot(unfold(core, mode), multikrocker(factors, transpose=True, skip = mode))
            U, S, V = partial_svd(unfold(tensor, mode).dot(W.T), n_eigenvecs = ranks[index])  #注意这里把取得奇异值的个数令为分解的秩了 ！
            factors[index] = U.dot(V)

            #update G
            prox_tensor = (mu*G[index] + Y[index] + tao*G[index])/(mu + tao)
            U_prox, S_prox, V_prox = partial_svd(prox_tensor)
            S_cut = np.zeros(prox_tensor.shape)
            for i, singular in enumerate(S_prox):
                if singular < 1/(mu+tao):
                    S_cut[i, i] = 0
                else:
                    S_cut[i, i] = S_prox[i] - 1/(mu+tao)
            mid = np.dot(U_prox,S_cut)
            G[index] = mid.dot(V_prox)

            #update Y
            Y[index] = Y[index] + Gamma*mu*(unfold(core, mode) - G[index])

        mu = mu + Rho * mu
        if mu > mu_max:
            mu = mu_max
        RSM = norm(unfold(core, len_ranks) - G[len_ranks-1], 2)
        print('Interation is', iteration,'RSM is', RSM)
        if RSM < Tol:
            break
    return core, factors

data = np.random.rand(100,100,10)
ranks = [10,10,10]
ADMM(data, ranks, 100)










