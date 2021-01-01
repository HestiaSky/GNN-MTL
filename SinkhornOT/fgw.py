import torch
import numpy as np
from .iterative_projection import gw_iterative_1, rgw_iterative_1, fgw_iterative_1, rfgw_iterative_1
from .cderivation import get_inter_sim, get_intra_sim, cos_dist_mat, p_norm_dist_mat
from .sinkhorn_loss import forward_relax_sinkhorn_iteration, sinkhorn_iteration
from UMH.rcsls_computation import get_rcsls_cost_matrix, rcsls_grad_respect_Q, our_rcsls_grad_respect_Q

def orthogonal_mapping_update(current_Q, gradient_Q, learning_rate):
    """
    update the Q with gradient and then project into orthogonal matrix
    :param current_Q:
    :param gradient_Q:
    :param learning_rate:
    :return:
    """
    next_Q = current_Q - learning_rate * gradient_Q
    try:
        u, s, v = torch.svd(next_Q)
    except:
        device = next_Q.device
        u, s, v = np.linalg.svd(next_Q.cpu().numpy())
        u = torch.from_numpy(u).to(device)
        s = torch.from_numpy(s).to(device)
        v = torch.from_numpy(v).to(device)
    return torch.mm(u, v.t())

def orthogonal_convex_hull_mapping_update(current_Q, gradient_Q, learning_rate):
    """
    update the Q with gradient and then project into orthogonal matrix
    :param current_Q:
    :param gradient_Q:our_rcsls_grad_respect_Q
    :param learning_rate:
    :return:
    """
    next_Q = current_Q - learning_rate * gradient_Q
    try:
        u, s, v = torch.svd(next_Q)
    except:
        device = next_Q.device
        u, s, v = np.linalg.svd(next_Q.cpu().numpy())
        u = torch.from_numpy(u).to(device)
        s = torch.from_numpy(s).to(device)
        v = torch.from_numpy(v).to(device)
    return torch.mm(u.mul(torch.clamp(s, max=1, min=0)), v.t())


def RFGW_mapping_gradient(X_all, Y_all, Q, batch_size, sample_space_size, alpha=0, p=2, max_iter=100, lambda_KL=1e-4, epsilon=1e-2, loss_type="L2", if_greedy_assignment = False,):
    """
    Calculate the gradient of the Q
    :param batch_X: m x d tensor on a unit ball
    :param batch_Y: m x d tensor on a unit ball
    :param Q: d x d tensor for mappingour_rcsls_grad_respect_Q
    :param alpha: the coefficient for fused GW
    :param lambda_KL: the coefficient for forward relaxation
    :return:
    """

    sample_slice = torch.randperm(len(X_all)).tolist()[:batch_size]
    batch_X = X_all[sample_slice,:]
    sample_slice = torch.randperm(len(Y_all)).tolist()[:batch_size]
    batch_Y = Y_all[sample_slice,:]
    X = torch.mm(batch_X, Q)
    Y = batch_Y

    device = X.device

    I = X.shape[0]
    J = Y.shape[0]

    mu = torch.ones(1, I, 1).to(device).double()
    nu = torch.ones(1, 1, J).to(device).double()

    Mt = -2 * X.mm(Y.t()).reshape(1, I, J).double()
    # mu = torch.ones(1, I, 1).to(device).double()
    # nu = torch.ones(1, 1, J).to(device).double()
    # if loss_type[:1] == 'L':
    #     p = int(loss_type[1:])


    #     Mt = -2 * torch.mm(X.t(), Y)

    # elif loss_type[:5] == "RCSLS":

    #     k = int(loss_type[5:])

    #     def dist_mat_func(x, y):
    #         return p_norm_dist_mat(x, y, p=2)

    # Mt = get_inter_sim(X, Y, cos_dist_mat).double().reshape(1, I, J)

    # else:
        # raise NotImplementedError

    if alpha > 0:
        C1t = get_intra_sim(X, cos_dist_mat).double()
        C2t = get_intra_sim(Y, cos_dist_mat).double()
        if lambda_KL > 0:
            T, _ = rfgw_iterative_1(Mt, C1t, C2t, mu, nu, alpha, p, max_iter, lambda_KL, epsilon)
        else:
            T, _ = fgw_iterative_1(Mt, C1t, C2t, mu, nu, alpha, p, max_iter, epsilon)
    else:
        if lambda_KL > 0:
            t, m1, m2, T = forward_relax_sinkhorn_iteration(Mt, mu, nu, lambda_KL, epsilon)
        else:
            t, m1, m2, T = sinkhorn_iteration(Mt, mu, nu, epsilon)

    T = T.float().reshape(I, J)
    if if_greedy_assignment:    
        result = torch.zeros_like(T)
        max_index = T.max(1)[1]
        for index in range(result.shape[0]):
            result[index][max_index[index]] = 1
        T = result


    if loss_type[:1] == "L":
        gradient = -2 * torch.mm(
                torch.mm(batch_X.t(), T.float().reshape(I, J)), 
                batch_Y)

        loss = -2 *  torch.trace(torch.mm(
            batch_X.mm(Q).t(), T.float().reshape(I, J).mm(batch_Y)
        ))
    elif loss_type[:5] == "RCSLS":
        k = int(loss_type[5:])
        loss, gradient = rcsls_grad_respect_Q(batch_X, batch_Y, X_all[:sample_space_size, :], Y_all[:sample_space_size, :], T, Q, k)

    else:
        raise NotImplementedError
    
    print("loss {:.4f} | marginal 1 {:.4f} | marginal 2 {:.4f} | sumT {:.4f} ".format(loss, m1, m2, T.sum()), end='\r')

    return gradient
