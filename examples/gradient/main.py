"""Validation of the gradient computation"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy as onp
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family":'serif',
    "font.size": 25,
    "lines.linewidth": 2.5
})

from examples.forward.main import jax_wrapper

if __name__ == "__main__":
    
    # Current loadings
    t_eval = (0,1000)
    dt = 5.
    c_rate = 2.
    
    # preparation
    fwd_pred, paramsets = jax_wrapper(t_eval, dt, c_rate=2.)
    theta0 = np.array([paramsets.alpha_an, paramsets.alpha_ca, paramsets.alpha_se,
                       paramsets.ks[0], paramsets.ks[1],
                       paramsets.tp,
                       paramsets.cs0_an, paramsets.cs0_ca])
    
    # Objective function
    def obj_fn(theta):
        sol_bs, _, _, _ = fwd_pred(theta)
        return np.sum(sol_bs)
    
    # Taylor test
    J0 = obj_fn(theta0)
    h_var = onp.array([1e-2,1e-3,1e-4,1e-5])
    res_0 = []
    res_1st = []
    for h in h_var:
        thetah = (1.+h) * theta0
        Jh, dJ = jax.value_and_grad(obj_fn)(thetah)
        # 0-order
        res_0.append(onp.array(onp.abs(Jh - J0)))
        # 1-st
        res_1st.append(onp.array(onp.abs(Jh - J0 - np.dot(dJ, h*theta0))))

    # postprocessing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output') 
    plt.figure(figsize=(10,8))
    # 0-order
    plt.plot(h_var, res_0, color='C0', marker='o', markersize = 10, label=r'$r_0$')
    plt.plot(h_var, h_var, linestyle='--', color='C0', label=r'1st order reference')
    # 1-st
    plt.plot(h_var, res_1st, color='C3', marker='o', markersize = 10, label=r'$r_1$')
    plt.plot(h_var, h_var**2, linestyle='--', color='C3', label=r'2nd order reference')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Residual')
    plt.xlabel(r'Step size $h$')
    plt.legend(frameon=False,fontsize=20)
    output_path = os.path.join(output_dir, 'taylor.png')
    plt.savefig(output_path, dpi=300, format="png",bbox_inches='tight')
    
    # Comparison with FDM
    # AD
    grad_ad = jax.grad(obj_fn)(theta0)
    
    # FDM
    h = 1e-3
    value_minus = obj_fn(theta0.at[0].add(-h))
    value_plus = obj_fn(theta0.at[0].add(h))
    grad_fd = (value_plus - value_minus)/(2.*h)
    
    print(f"grad_ad.shape = {grad_ad.shape}")
    print(f"grad_fd.shape = {grad_fd.shape}")
    print("grad_ad can give the gradient all at once, but grad_fd can only give one value!")
    print("Let's compare the value by both methods:")
    print(f"grad_ad[0] = {grad_ad[0]:.6e}")
    print(f"grad_fd = {grad_fd:.6e}")
    print(f"Relative error = {np.abs(grad_ad[0]-grad_fd)/np.abs(grad_fd)*100:.6f}%")
