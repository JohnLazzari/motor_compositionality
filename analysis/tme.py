import numpy as np
import torch
from scipy.linalg import svd
from scipy.optimize import minimize
from typing import List, Dict, Any, Tuple

def fit_max_entropy(params: Dict[str, Any]) -> Dict[str, Any]:
    marg_cov = params["margCov"]
    mean_tensor = params["meanTensor"]
    tensor_size = len(marg_cov)
    dim = list(mean_tensor.shape)
    
    eig_vectors = [None] * tensor_size
    eig_values = [None] * tensor_size
    tr_sigma = torch.zeros(tensor_size)

    # Compute shared trace
    shared_trace = None
    for i in range(tensor_size):
        if marg_cov[i] is not None:
            shared_trace = torch.trace(marg_cov[i])
            break

    for i in range(tensor_size):
        Sigma = marg_cov[i]
        if Sigma is None:
            Sigma = torch.eye(dim[i]) * (shared_trace / dim[i])
        # Use SVD as in MATLAB
        U, S_diag, _ = torch.linalg.svd(Sigma)
        S_sorted, indices = torch.sort(S_diag, descending=True)
        U_sorted = U[:, indices]
        eig_vectors[i] = U_sorted
        eig_values[i] = S_sorted
        tr_sigma[i] = torch.sum(S_sorted)

    if not torch.allclose(tr_sigma, torch.mean(tr_sigma).expand_as(tr_sigma), atol=(sum(dim) * torch.sqrt(torch.tensor(torch.finfo(torch.float64).eps)))):
        raise ValueError("The covariance matrices should have exactly the same trace.")

    # Low-rank handling
    threshold = -10
    pre_scale = eig_values[0].sum() / np.mean(dim)
    log_eig_values = []
    opt_dim = []

    for ev in eig_values:
        log_vals = torch.log(ev / pre_scale)
        valid_vals = log_vals[log_vals > threshold]
        log_eig_values.append(valid_vals)
        opt_dim.append(len(valid_vals))

    # Initialization of latent log-Lagrangians
    tensor_ixs = list(range(tensor_size))
    log_L0 = []
    for x in tensor_ixs:
        nx_set = [i for i in tensor_ixs if i != x]
        init = np.log(np.sum([opt_dim[i] for i in nx_set])) - log_eig_values[x].numpy()
        log_L0.extend(init)

    # Optimization
    max_iter = 10000

    def objective(logL_flat):
        return log_objective_max_entropy_tensor(torch.tensor(logL_flat), log_eig_values)

    res = minimize(
        fun=lambda x: objective(x).item(),
        x0=np.array(log_L0),
        method="L-BFGS-B",
        options={"maxiter": max_iter}
    )

    logL = res.x
    log_obj_per_iter = res.fun  # Not full history, just final
    L = np.exp(logL)
    lagrangians = []

    start = 0
    for x in tensor_ixs:
        stop = start + opt_dim[x]
        lagrangian = torch.tensor(np.concatenate([
            L[start:stop],
            [float("inf")] * (dim[x] - opt_dim[x])
        ])) / pre_scale
        lagrangians.append(lagrangian)
        start = stop

    # Final objective costs
    obj_cost = objective_max_entropy_tensor(torch.cat(lagrangians), eig_values).item()
    log_obj_cost = log_objective_max_entropy_tensor(torch.tensor(logL), log_eig_values).item()

    print(f' - final cost value = {log_obj_cost:.5e}')
    if obj_cost > 1e-5:
        print('Warning: Algorithm did not converge, results may be inaccurate')

    return {
        "meanTensor": mean_tensor,
        "eigVectors": eig_vectors,
        "Lagrangians": lagrangians,
        "objCost": obj_cost,
        "logObjperIter": log_obj_per_iter
    }




def diag_kron_sum(Ls: List[torch.Tensor]) -> torch.Tensor:
    """
    Constructs the Kronecker sum of diagonal matrices with given diagonal values.
    Returns a tensor of shape [prod(dim)].
    """
    result = Ls[0].view(-1, 1)
    for l in Ls[1:]:
        result = result + l.view(1, -1)
        result = result.flatten()
    return result




def sum_tensor(tensor: torch.Tensor, dims_to_sum: List[int]) -> torch.Tensor:
    """
    Sums a tensor over all dimensions in dims_to_sum.
    """
    for d in sorted(dims_to_sum, reverse=True):
        tensor = tensor.sum(dim=d, keepdim=True)
    return tensor.squeeze()




def log_objective_max_entropy_tensor(
    logL: torch.Tensor,
    log_eig_values: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates the log-transformed objective function and its gradient.

    Args:
        logL (Tensor): Stacked log Lagrangian eigenvalues (1D tensor).
        log_eig_values (list of Tensors): Each element is a 1D tensor of log-eigenvalues 
                                          for one marginal covariance matrix.

    Returns:
        f (Tensor): Scalar log-transformed objective value.
        gradf_logL (Tensor): Gradient of f w.r.t logL.
    """
    device = logL.device
    tensor_size = len(log_eig_values)
    dim = [len(le) for le in log_eig_values]
    cumdims = [0] + list(torch.cumsum(torch.tensor(dim), dim=0))

    logLs = [logL[cumdims[i]:cumdims[i+1]] for i in range(tensor_size)]
    Lagrangians = [logL_i.exp() for logL_i in logLs]

    normalize_term = torch.norm(torch.cat(log_eig_values))**2

    # Build Kronecker sum and its inverses
    Ls_tensor = diag_kron_sum(Lagrangians)  # shape: [∏ dim]
    Ls_tensor = Ls_tensor.reshape(dim)
    inv_Ls_tensor = 1.0 / Ls_tensor
    inv_sq_Ls_tensor = inv_Ls_tensor ** 2

    # Objective
    fx_list = []
    Er = []
    log_sums = []

    for i in range(tensor_size):
        other_dims = [j for j in range(tensor_size) if j != i]
        log_sum = torch.log(sum_tensor(inv_Ls_tensor, other_dims))
        log_sums.append(log_sum)
        err = log_eig_values[i].reshape(log_sum.shape) - log_sum
        Er.append(err)
        fx_list.append((err.view(-1) ** 2))

    f = torch.cat(fx_list).sum() / normalize_term

    # Gradient
    gradf_logL = torch.zeros(sum(dim), device=device)

    for i in range(tensor_size):
        other_dims = [j for j in range(tensor_size) if j != i]
        Er_i = Er[i]
        term1 = 2 * Er_i / sum_tensor(inv_Ls_tensor, other_dims)
        term1 *= sum_tensor(inv_sq_Ls_tensor, other_dims)
        gradfx_logLx = (term1.view(-1) * Lagrangians[i]).view(-1)

        gradfy_logLx = torch.zeros(dim[i], len(other_dims), device=device)

        for z, j in enumerate(other_dims):
            ny_set = [k for k in range(tensor_size) if k != j]
            nxy_set = [k for k in ny_set if k != i]
            Er_j = Er[j]

            term2 = 2 * Er_j / sum_tensor(inv_Ls_tensor, ny_set)
            term2 *= sum_tensor(inv_sq_Ls_tensor, nxy_set)
            grad = sum_tensor(term2, j)
            gradfy_logLx[:, z] = (grad.view(-1) * Lagrangians[i]).view(-1)

        gradf_block = gradfx_logLx + gradfy_logLx.sum(dim=1)
        gradf_logL[cumdims[i]:cumdims[i+1]] = gradf_block

    gradf_logL /= normalize_term
    return f, gradf_logL




def objective_max_entropy_tensor(
    L: torch.Tensor,
    eig_values: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate the maximum entropy tensor objective and its gradient.

    Args:
        L (Tensor): Stacked vector of Lagrangian eigenvalues.
        eig_values (list of Tensors): List of eigenvalue tensors from marginal covariances.

    Returns:
        f (Tensor): Objective cost function value.
        gradf_L (Tensor): Gradient of the objective w.r.t. L.
    """
    device = L.device
    tensor_size = len(eig_values)
    dim = [len(e) for e in eig_values]
    cumdims = [0] + list(torch.cumsum(torch.tensor(dim), dim=0))

    # Unstack Lagrangians
    Lagrangians = [L[cumdims[i]:cumdims[i+1]] for i in range(tensor_size)]
    normalize_term = torch.norm(torch.cat(eig_values))**2

    # Kronecker sum of diagonal Lagrangians
    Ls_tensor = diag_kron_sum(Lagrangians).reshape(dim)
    inv_Ls_tensor = 1.0 / Ls_tensor
    inv_sq_Ls_tensor = inv_Ls_tensor**2

    Er = []
    fx_list = []
    sums = []

    for i in range(tensor_size):
        other_dims = [j for j in range(tensor_size) if j != i]
        sum_term = sum_tensor(inv_Ls_tensor, other_dims)
        sums.append(sum_term)
        err = eig_values[i].reshape(sum_term.shape) - sum_term
        Er.append(err)
        fx_list.append((err.view(-1)**2))

    f = torch.cat(fx_list).sum() / normalize_term

    # Gradient computation
    gradf_L = torch.zeros(sum(dim), device=device)

    for i in range(tensor_size):
        other_dims = [j for j in range(tensor_size) if j != i]

        gradfx = 2 * Er[i] * sum_tensor(inv_sq_Ls_tensor, other_dims)
        gradfx_Lx = gradfx.view(-1)

        gradfy_Lx = torch.zeros(dim[i], len(other_dims), device=device)

        for z, j in enumerate(other_dims):
            nxy_set = [k for k in range(tensor_size) if k != i and k != j]
            grad_term = 2 * Er[j] * sum_tensor(inv_sq_Ls_tensor, nxy_set)
            summed = sum_tensor(grad_term, j)
            gradfy_Lx[:, z] = summed.view(-1)

        gradf_block = gradfx_Lx + gradfy_Lx.sum(dim=1)
        gradf_L[cumdims[i]:cumdims[i+1]] = gradf_block

    gradf_L /= normalize_term
    return f, gradf_L




def kron_mvprod(Qs: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """
    Multiplies x by the Kronecker product of matrices in Qs.

    Qs: list of orthogonal matrices [Q1, Q2, ..., Qn]
    x: tensor of shape (prod(dims), num_samples)

    Returns:
        Tensor of shape (prod(dims), num_samples)
    """
    out = x
    for i, Q in reversed(list(enumerate(Qs))):
        d = Q.shape[0]
        out = out.view(d, -1, out.shape[-1])
        out = torch.einsum("ij,jkl->ikl", Q, out)
        out = out.reshape(d * -1, out.shape[-1])
    return out





def sample_tme(max_entropy: Dict[str, any], num_surrogates: int = 1) -> torch.Tensor:
    """
    Generate tensor samples from the maximum entropy distribution
    with marginal mean and covariance constraints.

    Args:
        max_entropy (dict): Output from `fit_max_entropy` containing:
                            - 'Lagrangians': list of tensors (one per mode)
                            - 'eigVectors': list of tensors (one per mode)
                            - 'meanTensor': the mean tensor (not used in entropy, just added back)
        num_surrogates (int): Number of surrogate tensors to sample.

    Returns:
        Tensor: Surrogate tensor samples of shape (*dims, num_surrogates)
    """
    Lagrangians: List[torch.Tensor] = max_entropy["Lagrangians"]
    eig_vectors: List[torch.Tensor] = max_entropy["eigVectors"]
    mean_tensor: torch.Tensor = max_entropy["meanTensor"]

    tensor_size = len(eig_vectors)
    dim = [v.shape[0] for v in eig_vectors]  # mode sizes
    prod_dim = int(torch.tensor(dim).prod().item())

    # Inverse Kronecker sum of Lagrangians (diagonal case)
    D = 1.0 / diag_kron_sum(Lagrangians)

    # Sample from isotropic Gaussian
    x = torch.randn(prod_dim, num_surrogates, device=mean_tensor.device)
    x = D.sqrt().unsqueeze(1) * x  # scale by covariance eigenvalues

    # Multiply by eigenvectors
    x = kron_mvprod(eig_vectors, x)  # shape: (prod(dim), num_surrogates)

    # Add mean
    x = x + mean_tensor.reshape(-1, 1)

    # Reshape to full tensor
    surr_shape = tuple(dim) + (num_surrogates,)
    surr_tensors = x.reshape(*surr_shape)
    return surr_tensors