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

def ADMM(tensor, ranks, Lambda, max_iterations = 10000):

    #Initialize
    mu = 0.0001
    mu_max = 30000
    Rho = 1.03
    Tol = 1e-5
    Gamma = 1
    tao = 3*mu
    Y = []; G = []; factors = []
    rsm_CORE = []

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
        c = 0
        if iteration %50 == 0:
            c = c+1

        middle_re = np.zeros(ranks)

        reconstract_tensor = multi_mode_dot(core, factors, transpose=False)

        # print('iteration is ', iteration,'\n',  file=out)
        # print('mu is', mu, '\n', file=out)
        # print('1/mu is ', 1 / mu, '\n', file=out)
        # print('core', norm(core, 2),'\n', file = out)
        # print('reconstract_tensor ', norm(reconstract_tensor, 2), '\n', file=out)
        # print('tensor', norm(tensor, 2),'\n',  file = out)
        # print('middle_re', norm(middle_re, 2), '\n', file = out)

        # recalculate core
        for index in range(len_ranks):
            mode = index + 1
            middle_re += fold(G[index] - Y[index]/mu, mode, ranks)
        core = (Lambda/(Lambda + len_ranks*mu))*multi_mode_dot(tensor, factors, transpose=True) + (mu/(Lambda + len_ranks*mu))*middle_re

        #对每个factors, g, y 进行更新
        for index in range(len_ranks):
            mode = index + 1
            # update factors[n]
            W = multi_mode_dot(tensor, factors, skip=mode, transpose=True)
            U, S, V = partial_svd(unfold(W, mode).dot(unfold(middle_re, mode).T), n_eigenvecs = ranks[index])  #注意这里把取得奇异值的个数令为分解的秩了 ！
            factors[index] = U.dot(V)

            #update G
            prox_tensor = (mu*G[index] + Y[index] )/mu
            U_prox, S_prox, V_prox = partial_svd(prox_tensor)
            S_cut = np.zeros(prox_tensor.shape)

            """
            这里有问题，mu的选取对于对于奇异值根本没有任何的作用,应该是才用的图片的问题，换个低秩的图片加上噪声可能会有其他的效果
            """
            for i, singular in enumerate(S_prox):
                if singular < 1/mu :
                    S_cut[i, i] = 0
                else:
                    S_cut[i, i] = S_prox[i]
            G[index] = np.dot(U_prox,S_cut).dot(V_prox)

            #update Y
            Y[index] = Y[index] + Gamma*mu*(unfold(core, mode) - G[index])

        mu = Rho * mu
        if mu > mu_max:
            mu = mu_max
        tao = 3*mu


        RSM = norm(unfold(core, len_ranks) - G[len_ranks-1], 2)
        rsm_CORE.append(RSM)
        print('Interation is', iteration,'RSM is', RSM)
        RSM_tensor = norm(reconstract_tensor - tensor, 2)/norm(tensor, 2)
        print('RSM_tensor:', RSM_tensor)

        if iteration > 2:
            if abs(rsm_CORE[-1] - rsm_CORE[-2]) < Tol:
                break


    return core, factors

if __name__ == '__main__':
    out = open('record.txt', 'w')

    data = load_mat('Indian_pines.mat')
    ranks = [10, 10, 10]
    ADMM(data, ranks, 1000)




'''
ranks = [50, 50, 20]时 最小是1688961
ranks = [20,20,20]时， 最小是1839249
'''



'''
ranks = [10,10,10]
Interation is 3475 RSM is 0.00987282847693
RSM_tensor: 0.0743352843409
    #Initialize
    mu = 0.0001
    mu_max = 30000
    Rho = 1.03
    Tol = 1e-5
    Gamma = 1
    tao = 3*mu
    Y = []; G = []; factors = []
    lambda = 1000

'''




