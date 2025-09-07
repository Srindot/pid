import numpy as np

class Propeller:
    def get_ambient_conditions(self, h):
        P0, T0, g, M, R, L = 101325.0, 288.15, 9.80665, 0.0289644, 8.31447, 0.0065
        altitude = -h if h < 0 else 0
        temperature = T0 - L * altitude
        pressure = P0 * (1 - (L * altitude) / T0) ** ((g * M) / (R * L))
        density = (pressure * M) / (R * temperature)
        return pressure, density

class QuadcopterDynamics:
    def __init__(self, params, initial_state=None):
        self.m = params['m']
        self.g = params['g']
        self.Ixx = params['Ixx']
        self.Iyy = params['Iyy']
        self.Izz = params['Izz']
        self.Jr = params['Jr']
        self.l = params['l']
        self.h = params['h']
        self.state = np.zeros(12) if initial_state is None else np.array(initial_state, dtype=float)

    def get_derivatives(self, state, inputs):
        x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, p, q, r = state
        T, Q, H_x, H_y, R_mx, R_my = inputs['T'], inputs['Q'], inputs['H_x'], inputs['H_y'], inputs['R_mx'], inputs['R_my']
        Omega_r, Omega_r_dot = inputs['Omega_r'], inputs['Omega_r_dot']

        T_sum, Hx_sum, Hy_sum = sum(T), sum(H_x), sum(H_y)
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        
        sum_R_mx = R_mx[0] - R_mx[1] + R_mx[2] - R_mx[3]
        sum_R_my = R_my[0] - R_my[1] + R_my[2] - R_my[3]
        sum_Q = -Q[0] + Q[1] - Q[2] + Q[3]

        phi_ddot = (1/self.Ixx)*(q*r*(self.Iyy-self.Izz) + self.Jr*q*Omega_r + self.l*(-T[1]+T[3]) - self.h*Hy_sum + sum_R_mx)
        theta_ddot = (1/self.Iyy)*(p*r*(self.Izz-self.Ixx) - self.Jr*p*Omega_r + self.l*(T[0]-T[2]) + self.h*Hx_sum + sum_R_my)
        psi_ddot = (1/self.Izz)*(p*q*(self.Ixx-self.Iyy) + self.Jr*Omega_r_dot + self.l*(H_x[1]-H_x[3]) + self.l*(-H_y[0]+H_y[2]) + sum_Q)
        
        z_ddot = self.g - (c_phi*c_theta/self.m)*T_sum
        x_ddot = (1/self.m)*((s_phi*s_psi + c_phi*s_theta*c_psi)*T_sum - Hx_sum)
        y_ddot = (1/self.m)*((-s_phi*c_psi + c_phi*s_theta*s_psi)*T_sum - Hy_sum)
        
        return np.array([x_dot, y_dot, z_dot, p, q, r, x_ddot, y_ddot, z_ddot, phi_ddot, theta_ddot, psi_ddot])

    def update_state(self, dt, inputs):
        self.state += self.get_derivatives(self.state, inputs) * dt

class PD_Controller:
    def __init__(self, quad_params, target_z=-10.0, Kp_z=3.0, Kd_z=2.5):
        self.target_z = target_z
        self.Kp_z = Kp_z
        self.Kd_z = Kd_z
        self.hover_thrust = quad_params['m'] * quad_params['g']

    def calculate_outputs(self, state, t):
        z, z_dot = state[2], state[8]
        error_z = self.target_z - z
        error_z_dot = 0 - z_dot
        
        thrust_correction = self.Kp_z * error_z + self.Kd_z * error_z_dot
        total_thrust = np.clip(self.hover_thrust + thrust_correction, 0, self.hover_thrust * 2)
        
        thrust_per_motor = total_thrust / 4.0
        T = [thrust_per_motor] * 4
        
        return {
            "T": T, "Q": [0]*4, "H_x": [0]*4, "H_y": [0]*4, "R_mx": [0]*4, "R_my": [0]*4,
            "Omega_r": 0, "Omega_r_dot": 0
        }

class Simulator:
    def __init__(self, quad, controller):
        self.quad = quad
        self.controller = controller

    def run(self, duration=10.0, dt=0.01):
        num_steps = int(duration / dt)
        history = np.zeros((num_steps, len(self.quad.state)))
        
        for i in range(num_steps):
            current_time = i * dt
            inputs = self.controller.calculate_outputs(self.quad.state, current_time)
            self.quad.update_state(dt, inputs)
            history[i, :] = self.quad.state
        
        return history

if __name__ == '__main__':
    quad_params = {
        'm': 0.5, 'g': 9.81, 'Ixx': 0.0196, 'Iyy': 0.0196, 'Izz': 0.0264,
        'Jr': 0.0001, 'l': 0.25, 'h': 0.05
    }

    initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # Start at origin, on the ground (z=0)
    
    quad = QuadcopterDynamics(quad_params, initial_state=initial_state)
    controller = PD_Controller(quad_params, target_z=-10.0) # Target altitude of 10m (z=-10 in NED)
    sim = Simulator(quad, controller)

    print("Running simulation...")
    history = sim.run(duration=15.0, dt=0.01)
    
    final_state = history[-1]
    final_z = final_state[2]
    print(f"Simulation complete. Final altitude (z-position): {final_z:.2f} m")