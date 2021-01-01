import torch
import numpy as np
import ot
from SinkhornOT.sinkhorn_loss import sinkhorn_iteration
from SinkhornOT.cderivation import p_norm_dist_mat, get_init_matrices, get_LT, get_intra_sim, get_inter_sim, cos_dist_mat
from SinkhornOT.iterative_projection import *

dvc = "cuda:0" if torch.cuda.is_available() else "cpu"
I = 100
J = 100
D = 100
# np.random.seed(123)
a = np.ones((I,)) / I
b = np.ones((J,)) / J
T_init = np.ones((I, J)) / (I * J)
Va = np.random.uniform(size=(I, D))
Vb = np.random.uniform(size=(J, D))

Vat = torch.from_numpy(Va)
Vbt = torch.from_numpy(Vb)

Mt = get_inter_sim(Vat, Vbt, cos_dist_mat).double()
C1t = get_intra_sim(Vat, cos_dist_mat).double()
C2t = get_intra_sim(Vbt, cos_dist_mat).double()

M = Mt.numpy()
C1 = C1t.numpy()
C2 = C2t.numpy()

Mt = Mt.to(dvc).view(1, I, J)
at = torch.from_numpy(a).to(dvc).view(1, I, 1)
bt = torch.from_numpy(b).to(dvc).view(1, 1, J)
C1t = C1t.to(dvc)
C2t = C2t.to(dvc)


epsilon = 1e-4
niter = 100
thr = 1e-9


def diff_array_tensor(array, tensor):
    return np.max(abs(array - tensor))


def test_sinkhorn():
    T_pot = ot.sinkhorn(a, b, M, epsilon, method='sinkhorn_stabilized')
    W_pot = np.sum(np.multiply(np.asarray(T_pot), M))
    W_our, *_, T_our = sinkhorn_iteration(Mt, at, bt, epsilon)

    assert diff_array_tensor(W_pot, W_our) < thr
    assert diff_array_tensor(T_pot, T_our) < thr


def test_gw():

    T_pot, log_pot = ot.gromov.entropic_gromov_wasserstein(C1, C2, a, b,
                                                           loss_fun='square_loss',
                                                           epsilon=epsilon,
                                                           max_iter=100,
                                                           log=True)

    T_our, log_our = gromov_wasserstein_iterative1(C1t, C2t, at, bt,
                                                   epsilon=epsilon,
                                                   max_iter=100,
                                                   log=True)

    def log_compare(array_log: dict, tensor_log: dict):
        for k in array_log.keys():
            print("compare {}".format(k))
            if isinstance(array_log[k], list):
                for i, (a, t) in enumerate(zip(array_log[k], tensor_log[k])):
                    diff = diff_array_tensor(a, t)
                    if diff < thr:
                        print('safe', i, diff)
                    else:
                        print('danger !', i, diff)
            else:
                diff = diff_array_tensor(array_log[k], tensor_log[k])
                if diff < thr:
                    print("safe", diff)
                else:
                    print("danger", diff, array_log[k], tensor_log[k])

    log_compare(log_pot, log_our)
    T_diff = T_our.cpu().numpy() - T_pot
    T_diff /= np.max(T_pot)
    print(np.max(T_diff))
    print()


def test_runnable():
    alpha = 0.5
    p = 2
    lambdda = 1
    max_iter = 100
    # print()
    # gwd_our, T_our = gw_iterative_1(C1t, C2t, at, bt, max_iter, epsilon, False)
    # gwd_our, T_our = gw_iterative_2(C1t, C2t, at, bt, max_iter, epsilon, False)
    # print(gwd_our)
    T_our, gwd_our = fgw_iterative_1(Mt, C1t, C2t, at, bt, alpha, p, max_iter, epsilon, True)
    print(gwd_our)
    T_our, gwd_our = rgw_iterative_1(C1t, C2t, at, bt, max_iter, lambdda, epsilon, True)
    print(gwd_our)
    # gwd_our, T_our = rGW_iterative_projection(C1t, C2t, at, bt, max_iter, lambdda, epsilon, verbose=True)
    # print(gwd_our)
    # gwd_our, T_our = rFGW_iterative_projection(Mt, C1t, C2t, at, bt, alpha, p, max_iter, lambdda, epsilon, verbose=True)
    # print(gwd_our)


if __name__ == "__main__":
    test_runnable()
