"""Validation of the gradient computation"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import jax
import jax.numpy as np

from examples.forward.main import jax_wrapper

if __name__ == "__main__":
    
    # Current loadings
    t_eval = (0,3600)
    dt = 5.
    c_rate = 1.
    
    # preparation
    fwd_pred, paramsets = jax_wrapper(t_eval, dt, c_rate=1.)
    theta_0 = np.array([paramsets.alpha_an, paramsets.alpha_ca, paramsets.alpha_se,
                        paramsets.ks[0], paramsets.ks[1],
                        paramsets.tp,
                        paramsets.ds_an, paramsets.ds_ca,
                        paramsets.cs0_an, paramsets.cs0_ca])
    
    # Objective function
    def J_wrapper(ind):
        def J_total(theta_i):
            theta = theta_0.at[ind].set(theta_i)
            sol_bs_jax, _, _, _ = fwd_pred(theta)
            obj_val = sol_bs_jax[-1]
            return obj_val
        return J_total
    
    ind = 1
    J_total = J_wrapper(ind)
    
    # Gradinet computation
    # AD
    ad_grad = jax.grad(J_total)(theta_0[ind]) 
    
    # FDM
    eps = 1e-3
    theta_i_plus = theta_0[ind] * (1+eps)
    theta_i_minus = theta_0[ind] * (1-eps)
    fdm_grad = (J_total(theta_i_plus) - J_total(theta_i_minus))/(2*eps*theta_0[ind])
    
    print(f"\nAD grad = {ad_grad:.6e}\nFDM grad = {fdm_grad:.6e}")
    print(f"Relative error = {np.abs(ad_grad-fdm_grad)/np.abs(fdm_grad)*100:.6f}%")
    