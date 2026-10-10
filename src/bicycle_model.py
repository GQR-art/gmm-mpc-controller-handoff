import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

class DynamicBicycleModel:
    def __init__(self, params=None):
        self.params = params or {
            'm': 1500.0,
            'Iz': 2500.0,
            'a': 1.2,
            'b': 1.5,
            'Cf': 80000.0,
            'Cr': 80000.0,
        }
    
    def state_derivative(self, t, state, delta, vx):
        vy, r = state
        p = self.params
        m, Iz, a, b, Cf, Cr = p['m'], p['Iz'], p['a'], p['b'], p['Cf'], p['Cr']
        
        if abs(vx) < 0.1:
            vx = 0.1
        
        alpha_f = delta - (vy + a * r) / vx
        alpha_r = -(vy - b * r) / vx
        
        Fyf = Cf * alpha_f
        Fyr = Cr * alpha_r
        
        vy_dot = (Fyf + Fyr) / m - vx * r
        r_dot = (a * Fyf - b * Fyr) / Iz
        
        return [vy_dot, r_dot]
    
    def simulate(self, t, delta, vx, y0=None):
        if y0 is None:
            y0 = [0.0, 0.0]
        
        t_span = (t[0], t[-1])
        
        def delta_func(t_val):
            idx = np.argmin(np.abs(t - t_val))
            return delta[idx]
        
        def vx_func(t_val):
            idx = np.argmin(np.abs(t - t_val))
            return max(vx[idx], 0.1)
        
        def ode(t_val, state):
            d = delta_func(t_val)
            v = vx_func(t_val)
            return self.state_derivative(t_val, state, d, v)
        
        result = solve_ivp(ode, t_span, y0, t_eval=t, method='RK45')
        return result.y[0, :], result.y[1, :]

class ParameterIdentification:
    def __init__(self, features):
        self.f = features
        self.mask = features['vx'] > 1.0
        
    def identify_linear_regression(self):
        f = self.f
        m = self.mask
        
        vx = f['vx'][m]
        vy = f['vy'][m]
        r = f['r'][m]
        delta = f['delta'][m]
        ay = f['ay'][m]
        vy_dot = f['vy_dot'][m]
        r_dot = f['r_dot'][m]
        
        ay_eq = vy_dot + vx * r
        
        n = len(vx)
        X1 = np.column_stack([delta, r])
        y1 = ay_eq
        
        coeff1, _, _, _ = np.linalg.lstsq(X1, y1, rcond=None)
        k_ay1, k_ay2 = coeff1
        
        print(f"Linear Regression Results:")
        print(f"  ay = {k_ay1:.4f} * delta + {k_ay2:.4f} * r")
        
        X2 = np.column_stack([
            np.ones(n),
            r / vx,
            delta,
            delta / vx,
        ])
        y2 = ay_eq
        
        coeff2, _, _, _ = np.linalg.lstsq(X2, y2, rcond=None)
        print(f"  Detailed coeffs: {coeff2}")
        
        init_params = {
            'm': 1500.0, 'Iz': 2800.0, 'a': 1.15, 'b': 1.55,
            'Cf': 95000.0, 'Cr': 105000.0,
        }
        return init_params
        
    def cost_function(self, p_array):
        p_dict = {
            'm': p_array[0], 'Iz': p_array[1], 'a': p_array[2], 'b': p_array[3],
            'Cf': p_array[4], 'Cr': p_array[5],
        }
        
        model = DynamicBicycleModel(p_dict)
        f = self.f
        m = self.mask
        
        t = f['t'][m]
        delta = f['delta'][m]
        vx_data = f['vx'][m]
        vy_meas = f['vy'][m]
        r_meas = f['r'][m]
        
        vy0 = vy_meas[0]
        r0 = r_meas[0]
        
        try:
            vy_sim, r_sim = model.simulate(t, delta, vx_data, [vy0, r0])
            
            cost_vy = (vy_sim - vy_meas)
            cost_r = (r_sim - r_meas) * 5.0
            
            return np.concatenate([cost_vy, cost_r])
        except:
            return np.ones(2 * len(t)) * 1000.0
    
    def identify_nonlinear_optimization(self, init_params):
        lb = [1000.0, 1000.0, 0.5, 0.5, 20000.0, 20000.0]
        ub = [2500.0, 8000.0, 2.5, 2.5, 500000.0, 500000.0]
        
        p0 = [init_params['m'], init_params['Iz'], init_params['a'], init_params['b'],
              init_params['Cf'], init_params['Cr']]
        
        result = least_squares(
            self.cost_function, p0, bounds=(lb, ub), method='trf',
            loss='huber', f_scale=0.1, max_nfev=100, verbose=1,
        )
        
        p_opt = result.x
        return {
            'm': p_opt[0], 'Iz': p_opt[1], 'a': p_opt[2], 'b': p_opt[3],
            'Cf': p_opt[4], 'Cr': p_opt[5],
        }, result
    
    def compute_lateral_acceleration_based_params(self):
        f = self.f
        m = self.mask
        vx = f['vx'][m]
        r = f['r'][m]
        delta = f['delta'][m]
        ay = f['ay'][m]
        
        ay_reg = ay + vx * r
        
        for i in range(len(ay_reg)):
            if abs(ay_reg[i]) > 15:
                ay_reg[i] = 15 * np.sign(ay_reg[i])
        
        L = 2.7
        m_est = 1650
        
        X = delta - (L * r) / vx
        Y = ay_reg
        
        valid = np.abs(vx) > 3
        Xv, Yv = X[valid], Y[valid]
        
        p = np.polyfit(Xv, Yv, 1)
        slope = p[0]
        
        est_params = {
            'm': 1650.0,
            'Iz': 3000.0,
            'a': 1.3,
            'b': 1.4,
            'Cf': 75000.0,
            'Cr': 75000.0,
        }
        
        if slope > 0:
            C_combined = m_est * slope
            est_params['Cf'] = C_combined * 0.45
            est_params['Cr'] = C_combined * 0.55
        
        print(f"\nLateral Acceleration estimation:")
        print(f"  Linear fit slope: {slope:.2f}")
        
        return est_params
