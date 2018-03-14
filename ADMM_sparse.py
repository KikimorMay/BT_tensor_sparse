from pre import *
import numpy as np
import matplotlib.pyplot as plt

def to_image(tensor):
    """A convenience function to convert from a float dtype back to uint8"""
    im = tensor
    im -= im.min()
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)

def admm_sparse(tensor, Lambda, tau_sparse, Rho, modes, n_part, ranks, iteration=100, tol=1e-5):
    shape = tensor.shape
    sparse_tensor = np.zeros(shape)
    z_tensor = np.zeros(shape)
    Alpha = 0

    for iter in range(iteration):

        # _, _, tensor_block_recons = block_term_tensor_decomposition(tensor - sparse_tensor, modes=modes, n_part=n_part, ranks=ranks)
        _, _, partial_recons = partial_tucker(tensor - sparse_tensor, modes=modes, ranks=ranks)

        sparse_tensor = (Lambda * (tensor - partial_recons) + Rho * z_tensor + Alpha) / (Lambda + Rho)

        # update z_tensor:
        z_tensor = prox_Lasso(sparse_tensor - Alpha/Rho, tau_sparse/Rho)

        # update Lambda (这里可以加一个更新的步长
        Alpha += Rho * (z_tensor - sparse_tensor)

        err_total = (Lambda/2)*norm(tensor - partial_recons - sparse_tensor, 2)**2 + tau_sparse * norm(sparse_tensor, 1)
        if iter == 3:
            break

    return partial_recons, sparse_tensor


if __name__ == '__main__':
    image = np.array(imresize(face(), 0.1), dtype='float64')  # image has shape(768,1024,3)*0.3 = (230,307,3)
    a = np.random.randn(20, 20, 20)
    ranks = [100, 100, 3]
    # _, _, c = block_term_tensor_decomposition(image, modes=[1,2,3], n_part=2, ranks=ranks)
    _, _, b = partial_tucker(image, modes=[1,2,3], ranks=ranks)
    ans_except_sparse = admm_sparse(image, 100, tau_sparse=0.00001, Rho=1, modes=[1, 2, 3], n_part=2, ranks=ranks)

    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    ax.set_axis_off()
    ax.imshow(to_image(image))
    ax.set_title('original')

    ax = fig.add_subplot(1, 3, 2)
    ax.set_axis_off()
    ax.imshow(to_image(b))
    ax.set_title('partial_tucker')

    # ax = fig.add_subplot(1, 4, 3)
    # ax.set_axis_off()
    # ax.imshow(to_image(c))
    # ax.set_title('block_term')

    ax = fig.add_subplot(1, 3, 3)
    ax.set_axis_off()
    ax.imshow(to_image(ans_except_sparse))
    ax.set_title('except_sparse')
    plt.tight_layout()
    plt.show()