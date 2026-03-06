import numpy as np
import itertools
from sklearn.linear_model import LassoCV

def solve_surrogate_system(X, Y, Y_dx, basis_funcs, deriv_funcs):
    N, d = X.shape
    p = len(basis_funcs) - 1

    multi_index = [
        alpha for alpha in itertools.product(range(p + 1), repeat=d) 
        if sum(alpha) <= p
    ]
    card = len(multi_index)

    def assemble_block(d_idx):
        Psi_block = np.zeros((N, card))
        for i in range(N):
            for j, alp in enumerate(multi_index):
                prod = 1.0
                for m in range(d):
                    if m == d_idx:
                        prod *= deriv_funcs[alp[m]](X[i, m])
                    else:
                        prod *= basis_funcs[alp[m]](X[i, m])
                Psi_block[i, j] = prod
        return Psi_block

    Psi = assemble_block(-1)
    all_psi = [Psi] + [assemble_block(k) for k in range(d)]
    Psi_comb = np.concatenate(all_psi, axis=0)
    Y_comb = np.concatenate([Y] + [Y_dx[:, i] for i in range(d)])

    n_total_obs = len(Y)
    lasso = LassoCV(cv=n_total_obs, fit_intercept=False, max_iter=20000)
    lasso.fit(Psi, Y)
    
    n_total_obs_comb = len(Y_comb)
    lasso_comb = LassoCV(cv=n_total_obs_comb, fit_intercept=False, max_iter=20000)
    lasso_comb.fit(Psi_comb, Y_comb)
    
    coef_mapping = dict(zip(multi_index, lasso.coef_))
    coef_mapping_comb = dict(zip(multi_index, lasso_comb.coef_))
    
    return {
        'coefficients': dict(zip(multi_index, lasso.coef_)),
        'grad_enhanced_coefficients': dict(zip(multi_index, lasso_comb.coef_)),
        'matrix': Psi,
        'large_matrix': Psi_comb,
        'multi_indices': multi_index,
    }

def matrix_only(X, basis_funcs, deriv_funcs):
    N, d = X.shape
    p = len(basis_funcs) - 1

    multi_index = [
        alpha for alpha in itertools.product(range(p + 1), repeat=d) 
        if sum(alpha) <= p
    ]
    card = len(multi_index)

    def assemble_block(d_idx):
        Psi_block = np.zeros((N, card))
        for i in range(N):
            for j, alp in enumerate(multi_index):
                prod = 1.0
                for m in range(d):
                    if m == d_idx:
                        prod *= deriv_funcs[alp[m]](X[i, m])
                    else:
                        prod *= basis_funcs[alp[m]](X[i, m])
                Psi_block[i, j] = prod
        return Psi_block

    Psi = assemble_block(-1)
    all_psi = [Psi] + [assemble_block(k) for k in range(d)]
    Psi_comb = np.concatenate(all_psi, axis=0)

    return Psi_comb

import numpy as np

def surrogate_constructor(basis_funcs, deriv_funcs, coefficients, deriv_idx=-1):
    
    sample_key = next(iter(coefficients))
    d = len(sample_key)
    
    active_indices = {alp: c for alp, c in coefficients.items() if c != 0}

    def model(X):
        X = np.atleast_2d(X)   # ensures shape (n, d)
        results = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            val = 0.0
            for alp, c in active_indices.items():
                prod = 1.0
                for m in range(d):
                    if m == deriv_idx:
                        if alp[m] > 0:
                            prod *= deriv_funcs[alp[m]](x[m])
                        else:
                            prod = 0.0
                            break
                    else:
                        prod *= basis_funcs[alp[m]](x[m])
                val += c * prod
            results[i] = val

        return results if len(results) > 1 else results[0]

    return model


def compute_sobol_indices(coefficients):

    # Values extraction
    all_coeffs = np.array(list(coefficients.values()))
    
    # First multi-index. To exclude it to compute the variance
    constant_multi = tuple(0 for _ in next(iter(coefficients)))
    d = len(constant_multi)
    
    # Variance
    squared_coeffs = {alp: c**2 for alp, c in coefficients.items() if alp != constant_multi} # Store all the squared coefficients
    V = sum(squared_coeffs.values())

    first_order = []
    total_order = []

    for i in range(d):
        # First-order indices
        # Take the sum of the coefficients for which ONLY the variable x_i is incolved
        v_i = sum(
            c_sq for alp, c_sq in squared_coeffs.items()
            if alp[i] > 0 and sum(alp) == alp[i]
        )
        first_order.append(v_i / V)

        # Total Sobol indices
        # Take the sum of the coefficients for which the variable x_i is involved
        v_ti = sum(
            c_sq for alp, c_sq in squared_coeffs.items()
            if alp[i] > 0
        )
        total_order.append(v_ti / V)

    return {
        "First_order": np.array(first_order),
        "Total": np.array(total_order),
        "Variance": np.array(V)
    }