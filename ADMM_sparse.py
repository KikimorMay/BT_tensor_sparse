from pre import *
import numpy as np


def admm_sparse(tensor, Lambda, tau_sparse, Rho, modes, n_part, ranks, iteration=100, tol=1e-5):
    shape = tensor.shape
    sparse_tensor = np.zeros(shape)
    z_tensor = np.zeros(shape)
    Alpha = 0

    for iter in range(iteration):

        _, _, tensor_block_recons = block_term_tensor_decomposition(tensor - sparse_tensor, modes=modes, n_part=n_part, ranks=ranks)
        _, _, partial_recons = partial_tucker(tensor - sparse_tensor, modes=modes, ranks=ranks)

        sparse_tensor = (Lambda * (tensor - partial_recons) + Rho * z_tensor + Alpha) / (Lambda + Rho)

        # update z_tensor:
        z_tensor = prox_Lasso(sparse_tensor - Alpha/Rho, tau_sparse/Rho)

        # update Lambda (这里可以加一个更新的步长
        Alpha += Rho * (z_tensor - sparse_tensor)

        err_total = (Lambda/2)*norm(tensor - tensor_block_recons - sparse_tensor, 2)**2 + tau_sparse * norm(sparse_tensor, 1)
        print('sparse_tensor:', norm(sparse_tensor, 2)/norm(tensor, 2))
        print('err_tatal:', err_total)


    return


if __name__ == '__main__':
    image = np.array(imresize(face(), 0.1), dtype='float64')  # image has shape(768,1024,3)*0.3 = (230,307,3)
    a = np.random.randn(20, 20, 20)
    ranks = [3, 3, 3]
    print(norm(a, 2))
    _, _, c = block_term_tensor_decomposition(a, modes=[1,2,3], n_part=2, ranks=ranks)
    _, _, b = partial_tucker(a, modes=[1,2,3], ranks=ranks)
    admm_sparse(a, 100, tau_sparse=0.00001, Rho=1, modes=[1, 2, 3], n_part=2, ranks=ranks)
