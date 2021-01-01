import torch
import numpy as np


def sinkhorn(a, b, M, reg, numItermax = 1000, stopThr = 1e-9, verbose = False):

    """
    Solve the entropic regularization balanced optimal transport problem 

    Parameters:
    param: a(tensor (I, )) sample weights for source measure
    param: b(tensor (J, )) sample weights for target measure
    param: M(tensor (I, J)) distance matrix between source and target measure
    param: reg(float64) regularization factor > 0
    param: numItermax(int) max number of iterations
    param: stopThr(float64) stop threshol
    param: verbose(bool) print information along iterations

    Return:
    P(tensor (I, J)) the final transport plan
    loss(float) the wasserstein distance between source and target measure
    """
    import time
    assert a.device == b.device and b.device == M.device, "a, b, M must be on the same device"

    device = a.device
    a, b, M = a.type(torch.DoubleTensor).to(device), b.type(torch.DoubleTensor).to(device), M.type(torch.DoubleTensor).to(device)

    if len(a) == 0:
        a = torch.ones(M.shape[0], dtype=torch.DoubleTensor) / M.shape[0]
    if len(b) == 0:
        b = torch.ones(M.shape[1], dtype=torch.DoubleTensor) / M.shape[1]
    
    I, J = len(a), len(b)
    assert I == M.shape[0] and J == M.shape[1], "the dimension of weights and distance matrix don't match"

    # init 
    u = torch.ones((I, 1), device = device, dtype=a.dtype) / I
    v = torch.ones((J, 1), device = device, dtype=b.dtype) / J
    # K = torch.exp(-M / reg).to(device)
    K = torch.empty(M.size(), dtype=M.dtype, device=device)
    torch.div(M, -reg, out=K)
    torch.exp(K, out=K)

    tmp2 = torch.empty(b.shape, dtype=b.dtype, device=device)

    Kp = (1 / a).reshape(-1, 1) * K
    cpt, err = 0, 1 
    # pos = time.time()
    while (err > stopThr and cpt < numItermax):
        uprev, vprev = u, v

        KtranposeU = torch.mm(K.t(), u)
        v = b.reshape(-1, 1) / KtranposeU
        u = 1. / Kp.mm(v)

        if (torch.any(KtranposeU == 0)
                or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
            print("Warning: numerical errors at iteration ", cpt)
            u, v = uprev, vprev
            break
        
        if cpt % 10 == 0:
            tmp2 = torch.einsum('ia,ij,jb->j', u, K, v)
            err = torch.norm(tmp2 - b)
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:5s}'.format('It.','Err') + '\n' + '-' * 19)
                print("{:5s}|{:5s}".format(cpt, err))
        
        cpt += 1
    # print("ours cpt: {}, err: {}".format(cpt, err))
    # print("ours time: {}".format(time.time() - pos))
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    return P, torch.sum(P * M)


if __name__ == "__main__":
    n, m = 20, 20
    a, b = np.ones(20) / 20, np.ones(20) / 20
    X, Y = np.random.randn(n, 10), np.random.randn(m, 10)
    import scipy
    import ot
    C = scipy.spatial.distance.cdist(X, Y, metric="euclidean")
    Tarr = ot.sinkhorn(a, b, C, reg=0.01) 

    a_ten, b_ten = torch.from_numpy(a), torch.from_numpy(b)
    C_ten = torch.from_numpy(C)
    T_ten, _ = sinkhorn(a_ten, b_ten, C_ten, reg=0.01)

    err = torch.norm(T_ten - torch.from_numpy(Tarr))

    print("erorr between Tarr and Tten: {:.4f}".format(err))