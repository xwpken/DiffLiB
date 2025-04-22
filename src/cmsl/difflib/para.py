# Copyright (C) 2025 DiffLiB authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
"""Model parameter sets."""

from dataclasses import dataclass

import jax.numpy as np

@dataclass
class ParameterSets:
    
    dt:float                    # time step size
    c_rate:float                # c-rate
    l_ref:float = 1e-6          # characteristic length
    
    def __post_init__(self):
        
        # ---- geometry ----
        
        self.l = [25e-6, 100e-6, 25e-6, 100e-6, 25e-6]
        
        self.W_cell = 0.137
        self.H_cell = 0.207
        
        
        # ---- physical constants ----
        
        self.F = 96485.33212       # Faraday's constant (C/mol)
        self.R = 8.314462618       # Gas constant
        self.T = 298.15            # Absolute temperature 25 celsius
        
        
        # ---- micro ----
        
        # anode particle
        
        self.r_an = 10e-6 # (m) particle radius in anode
        self.l_ref_an = self.r_an
        
        self.ds_an = 3.9e-14 # m^2/s
        
        # cathode particle
        
        self.r_ca = 10e-6 # (m) particle radius in cathode
        self.l_ref_ca = self.r_ca
        
        self.ds_ca = 1e-13 # m^2/s
        
        
        # ---- macro ----
        
        # volume fraction
        self.epss_an = 0.60                             # active particles
        self.epsl_an = 0.30                             # electrolyte
        self.epsf_an = 1 - self.epss_an - self.epsl_an  # filler
        
        self.epss_ca = 0.50                             # active particles
        self.epsl_ca = 0.30                             # electrolyte
        self.epsf_ca = 1 - self.epss_ca - self.epsl_ca  # filler
        
        self.epsl_se = 1.00                             # electrolyte
        
        # bruegmann coefficient
        self.alpha_an = 1.5
        self.alpha_ca = 1.5
        self.alpha_se = 1.5
        
        # Initial Li+ in electrolyte (mol/m^3)
        self.cl0 = 1000
        self.cl_ref = self.cl0
        
        # ---- conduction of potential in electrolyte ----
        
        # ionic conductivity
        self.kappa = self.calcKappa(self.cl0)
        self.ka_ref = self.kappa
        
        # coefficient for current source term
        self.sour_p = (self.l_ref)**2 / self.ka_ref * self.F
        
        
        # ---- diffusion of Li+ in electrolyte ----
        
        # diffusivity
        # self.df = 2.7877e-10
        self.df = self.calcDf(self.cl0)
        self.df_ref = self.df
        
        # coefficient for time diff term
        self.c_dt_coeff = 1 / self.dt * self.l_ref**2 / self.df_ref
        
        # transference number
        self.tp = 0.4
        
        # coefficient for species source term
        self.sour_c = (self.l_ref)**2 / self.df_ref * (1 - self.tp)
        
        
        # ---- conduction of potential in electrode ----
        
        self.sigma_ref = 10
        
        self.I_bc_coeff = self.l_ref/ self.sigma_ref
        
        # electrode
        
        # anode
        self.sigma_an = 100. # (s/m)
        self.sour_s_an = (self.l_ref)**2 / self.sigma_ref * self.F
        
        # cathode
        self.sigma_ca = 10.  # (s/m)
        self.sour_s_ca = (self.l_ref)**2 / self.sigma_ref * self.F
        
        # acc
        self.sigma_acc = 59600000.0 # (s/m)  # pybamm default
        
        # ccc
        self.sigma_ccc = 35500000.0 # (s/m)  # pybamm default
        
        
        # ---- interfacial kinetics (BV) ----
        
        self.j_ref = 1e-4
        
        self.ks = [2e-5/self.F, 6e-7/self.F] # m^2.5 mol^-0.5 s^-1
        
        # maximum Li+ in electrode (mol/m^3)
        self.cs_max = [2.4983e+04, 5.1218e+04]
        
        # initial Li+ in electrode (mol/m^3)
        self.cs0_an = 19987
        self.cs0_ca = 30731
        
        # ---- applied external current (1C = 0.680616[A]) ----
        self.I_bc = self.c_rate * 24 # 1C/(0.207*0.137) (A/m^2) 
        
        self.cut_off = 3.105
        
        
    def calcKappa(self, c):
        
        a0 = 0.0911;
        a1 = 1.9101e-3;
        a2 = -1.052e-6;
        a3 = 0.1554e-9;
        
        kappa = a0 + a1 * c + a2 * c**2 + a3 * c**3;
        
        return kappa


    def calcDf(self, c_e):

        return 5.34e-10 * np.exp(-0.65 * c_e / 1000)


    def calcUoc_neg(self, sto):
        
        exp = np.exp
        tanh = np.tanh
        
        Uoc_an = (0.194 + 1.5 * exp(-120.0 * sto)
                +0.0351 * tanh((sto - 0.286) / 0.083) 
                -0.0045 * tanh((sto - 0.849) / 0.119) 
                -0.035 * tanh((sto - 0.9233) / 0.05) 
                -0.0147 * tanh((sto - 0.5) / 0.034) 
                -0.102 * tanh((sto - 0.194) / 0.142) 
                -0.022 * tanh((sto - 0.9) / 0.0164) 
                -0.011 * tanh((sto - 0.124) / 0.0226) 
                +0.0155 * tanh((sto - 0.105) / 0.029))
        
        return Uoc_an


    def calcUoc_pos(self, sto):
        
        sto = sto * 1.062
        
        tanh = np.tanh
        
        Uoc_ca = (2.16216 + 0.07645 * tanh(30.834 - 54.4806 * sto) 
               +2.1581 * tanh(52.294 - 50.294 * sto) 
               -0.14169 * tanh(11.0923 - 19.8543 * sto) 
               +0.2051 * tanh(1.4684 - 5.4888 * sto) 
               +0.2531 * tanh((-sto + 0.56478) / 0.1316) 
               -0.02167 * tanh((sto - 0.525) / 0.006))
        
        return Uoc_ca