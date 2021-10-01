# xxx deprecate
def posn2vec(M, posn, tau=None):
    """
    Discrete position to column of matrix by look-up,
    or soft position to column of matrix via attention
    """
    if tau is None:
        return M[:,posn]
    attn = posn2attn(posn, tau, M.shape[-1])
    return attn2vec(attn)


# xxx deprecate
def bposn2vec(M, posn, tau=None):
    """
    Discrete position to column of matrix by look-up,
    or soft position to column of matrix via attention
    note: discrete position must be torch.LongTensor,
    soft position must be torch.FloatTensor
    => output is nbatch x nrow(M)
    """
    if tau is None:
        return M[:,posn].t()
    attn = bposn2attn(posn, tau, M.shape[-1])
    val = battn2vec(M, attn)
    return val
    # example with discrete positions:
    # posn = torch.LongTensor(np.array([0,3,5]))
    # u = bposn2vec(U, posn)
    # print(u.data)


# xxx deprecate
def attn2vec(M, attn):
    """
    Attention distribution to column of matrix
    """
    val = torch.mm(M, attn)
    return val


# xxx deprecate
def battn2vec(M, attn):
    """
    Attention distribution to column of matrix
    """
    val = torch.mm(M, attn.t()).t()
    #val = torch.einsum('bj,ij->bi', [M, attn])
    return val


# xxx deprecate
def posn2role(posn, tau=None):
    """
    Soft position to soft role vector
    """
    R = config.R
    return posn2vec(R, posn, tau)


# xxx deprecate
def bposn2role(posn, tau=None):
    """
    Soft position to soft role vector,
    output is nbatch x drole
    """
    R = config.R
    return bposn2vec(R, posn, tau)


# xxx deprecate
def battn2role(attn):
    R = config.R
    return battn2vec(R, attn)


# xxx deprecate
def battn2succ(attn):
    return battn2vec(S, attn)


# xxx deprecate
def posn2unbind(posn, tau=None):
    """
    Soft position to soft unbinding vector
    """
    U = config.U
    return posn2vec(U, posn, tau)


# xxx deprecate
def bposn2unbind(posn, tau=None):
    """
    Soft position to soft unbinding vector,
    output is nbatch x drole
    """
    U = config.U
    return bposn2vec(U, posn, tau)


# xxx deprecate
def battn2unbind(attn):
    """
    Attention distribution to soft unbinding vector
    """
    U = config.U
    return battn2vec(U, attn)

# xxx used by decoder
def posn2filler(T, posn, tau=None):
    """
    Unbind filler at hard or soft string position
    """
    u = posn2unbind(posn, tau)
    f = T.mm(u)
    return f


# xxx deprecate
def bposn2filler(T, posn, tau=None):
    """
    Unbind filler at hard or soft string position
    for each position i in batch posn
    - get unbinding vector ui by look-up or attention
    - unbind filler fi from tpr Ti with ui
    output is nbatch x nfill
    """
    u = bposn2unbind(posn, tau)
    #print(T.shape, u.shape)
    f = T.bmm(u.unsqueeze(2))
    #print(f.shape)
    f = f.squeeze(-1)
    #print(f.shape)
    return f


# xxx deprecate
def battn2filler(T, attn):
    u = battn2vec(config.U, attn)
    f = T.bmm(u.unsqueeze(2))
    f = f.squeeze(-1)
    return f


# xxx deprecate
def bbind(f, r):
    """
    Contruct filler-role bindings,
    output is nbatch x nfill x nrole
    [https://discuss.pytorch.org/t/batch-outer-product/4025/2]
    """
    T = torch.bmm(f.unsqueeze(2), r.unsqueeze(1))
    return T


def normalize(X, dim=1):
    """
    Normalize by summing over second dimension
    (NB. does not take absolute value of elements,
    therefore not equivalent to torch.normalize(1,1)).
    todo: relocate
    """
    Y = X / torch.sum(X, dim, keepdim=True)
    return Y


# xxx deprecate
def normalize_length(X, dim=1):
    """
    Normalize to length one.
    todo: relocate
    """
    Y = X / torch.mm(X.t(), X)
    return Y


# xxx deprecate
def bbound(X):
    """
    Bound columns of each batch within [-1,1].
    todo: relocate
    """
    batch_size, m, n = X.shape
    ones = torch.ones((batch_size,1,n))
    maxi = torch.max(X, 1)[0].view(batch_size, 1, n)
    mini = torch.min(X,1)[0].view(batch_size, 1, n)
    maxi = torch.cat((maxi, -mini, ones), 1)
    maxi = torch.max(maxi, 1)[0].view(batch_size, 1, n)
    Y = X / maxi
    #delta = torch.sum(torch.sum(Y - brescale(X), 0), 0)
    #print(delta.data[0])
    return Y


# xxx deprecate
def check_bounds(x, min=0.0, max=1.0):
    """
    Check that each value of vector is within bounds.
    todo: relocate
    """
    if np.any(x<min) or np.any(x>max):
        return 0
    return 1