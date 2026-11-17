"""EKF motion and measurement models for Lab 04, Tasks 1 and 2.

Task 1 uses [x, y, theta]; Task 2 uses [x, y, theta, v, omega].
Both variants are retained in RobotEKF to make comparison straightforward.
"""

import numpy as np


def normalize_angle(angle):
    """Wrap a heading/bearing to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def residual_landmark(z, z_hat):
    """Landmark innovation: range difference, wrapped bearing difference."""
    residual = np.asarray(z, dtype=float) - np.asarray(z_hat, dtype=float)
    residual[1] = normalize_angle(residual[1])
    return residual


def residual_vector(z, z_hat):
    """Ordinary innovation for linear and angular velocity measurements."""
    return np.asarray(z, dtype=float) - np.asarray(z_hat, dtype=float)


def _sinc_and_derivative(a):
    """Return sin(a)/a and its derivative, stable near a=0."""
    if abs(a) < 1e-4:
        a2 = a * a
        sinc = 1.0 - a2 / 6.0 + a2 * a2 / 120.0
        dsinc = -a / 3.0 + a * a2 / 30.0 - a * a2 * a2 / 840.0
        return sinc, dsinc
    return np.sin(a) / a, (a * np.cos(a) - np.sin(a)) / (a * a)


def _velocity_increment(theta, v, w, dt):
    """Exact constant-turn-rate translation, including the w=0 limit."""
    half_turn = 0.5 * w * dt
    sinc, _ = _sinc_and_derivative(half_turn)
    direction = theta + half_turn
    return v * dt * sinc * np.array([np.cos(direction), np.sin(direction)])


# ----------------------- Task 1: three-state motion ----------------------- #

def eval_gux(mu, u, sigma_u, dt):
    """Three-state velocity motion model, x' = g(x, [v, omega])."""
    del sigma_u  # Noise is propagated through the EKF covariance, not sampled.
    x, y, theta = mu
    v, w = u
    dx, dy = _velocity_increment(theta, v, w, dt)
    return np.array([x + dx, y + dy, normalize_angle(theta + w * dt)])


def eval_Gt(x, y, theta, v, w, dt):
    """Task 1 Jacobian dg/d[x,y,theta], shape (3,3)."""
    del x, y
    dx, dy = _velocity_increment(theta, v, w, dt)
    return np.array([
        [1.0, 0.0, -dy],
        [0.0, 1.0,  dx],
        [0.0, 0.0, 1.0],
    ])


def eval_Vt(x, y, theta, v, w, dt):
    """Task 1 Jacobian dg/d[v,omega], shape (3,2).

    The zero-angular-velocity limit includes the second-order heading effect:
    d(x,y)/domega = (-v*dt^2*sin(theta)/2, v*dt^2*cos(theta)/2).
    """
    del x, y
    a = 0.5 * w * dt
    sinc, dsinc = _sinc_and_derivative(a)
    phi = theta + a
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    return np.array([
        [dt * sinc * cos_phi,
         0.5 * v * dt**2 * (dsinc * cos_phi - sinc * sin_phi)],
        [dt * sinc * sin_phi,
         0.5 * v * dt**2 * (dsinc * sin_phi + sinc * cos_phi)],
        [0.0, dt],
    ])


# ---------------------- Task 2: five-state motion ------------------------ #

def eval_gux_5(mu, u, sigma_u, dt):
    """Constant-velocity 5D state transition.

    mu = [x, y, theta, v, omega].  Velocities are states, not inputs:
    u is deliberately unused to avoid feeding odometry in twice.
    """
    del u, sigma_u
    pose_next = eval_gux(mu[:3], mu[3:5], None, dt)
    return np.concatenate((pose_next, np.asarray(mu[3:5], dtype=float)))


def eval_Gt_5(x, y, theta, v, w, dt):
    """Jacobian dg_5/d[x,y,theta,v,omega], shape (5,5).

    Upper-left block is the Task 1 G; upper-right is the Task 1 V.
    """
    G = np.eye(5)
    G[:3, :3] = eval_Gt(x, y, theta, v, w, dt)
    G[:3, 3:5] = eval_Vt(x, y, theta, v, w, dt)
    return G


def eval_Vt_5(x, y, theta, v, w, dt):
    """Jacobian wrt velocity *process-noise increments*, shape (5,2).

    During one prediction interval, noisy increments in v and omega affect
    the travelled pose as well as the two retained velocity state variables.
    Q_process = V_5 @ M @ V_5.T is thus a 5-by-5 covariance.
    """
    V = np.zeros((5, 2))
    V[:3, :] = eval_Vt(x, y, theta, v, w, dt)
    V[3:, :] = np.eye(2)
    return V


# ----------------------- Landmark model (both) -------------------------- #

def eval_hx(mu, landmark_pos):
    """Expected [range (m), bearing (rad)] for a landmark."""
    x, y, theta = mu[:3]
    lx, ly = landmark_pos
    dx, dy = lx - x, ly - y
    return np.array([
        np.hypot(dx, dy),
        normalize_angle(np.arctan2(dy, dx) - theta),
    ])


def eval_Ht(mu, landmark_pos):
    """Landmark Jacobian: 2x3 for Task 1, 2x5 for Task 2.

    Range/bearing do not directly measure velocity, so the last two columns
    of the Task 2 Jacobian are zero.
    """
    x, y = mu[:2]
    lx, ly = landmark_pos
    dx, dy = lx - x, ly - y
    q = dx * dx + dy * dy
    H = np.zeros((2, len(mu)))
    if q < 1e-10:
        return H  # Avoid division by zero when robot and landmark coincide.
    r = np.sqrt(q)
    H[:, :3] = [
        [-dx / r, -dy / r, 0.0],
        [ dy / q, -dx / q, -1.0],
    ]
    return H


# ------------------------ Task 2: new sensors --------------------------- #

def eval_hx_odom(mu):
    """Expected wheel-odometry measurement [linear_v, angular_w]."""
    return np.asarray(mu[3:5], dtype=float).copy()


def eval_Ht_odom(mu):
    """Wheel-odometry measurement Jacobian, shape (2,5)."""
    H = np.zeros((2, len(mu)))
    H[0, 3] = 1.0
    H[1, 4] = 1.0
    return H


def eval_hx_imu(mu):
    """Expected IMU yaw-rate measurement [angular_w]."""
    return np.array([mu[4]], dtype=float)


def eval_Ht_imu(mu):
    """IMU yaw-rate measurement Jacobian, shape (1,5)."""
    H = np.zeros((1, len(mu)))
    H[0, 4] = 1.0
    return H


def sample_odom_measurement(mu, std_odom, rng=None):
    """Simulated [v,omega] observation with independent Gaussian noise."""
    rng = np.random.default_rng() if rng is None else rng
    return eval_hx_odom(mu) + rng.normal(0.0, std_odom, size=2)


def sample_imu_measurement(mu, std_imu, rng=None):
    """Simulated IMU yaw-rate observation (one-element array)."""
    rng = np.random.default_rng() if rng is None else rng
    return eval_hx_imu(mu) + rng.normal(0.0, std_imu, size=1)


class RobotEKF:
    """Extended Kalman filter for either 3D (Task 1) or 5D (Task 2) state."""

    def __init__(self, dim_x=3, dim_u=2):
        if dim_x not in (3, 5):
            raise ValueError('RobotEKF supports 3- or 5-element state vectors.')
        if dim_u != 2:
            raise ValueError('Velocity process noise must have two components.')

        self.mu = np.zeros(dim_x)
        self.Sigma = np.eye(dim_x) * 0.1
        self.Mt = np.eye(dim_u)  # Process covariance for [delta_v,delta_omega]
        self._I = np.eye(dim_x)
        self.dim_x = dim_x

        if dim_x == 5:
            self.eval_gux = eval_gux_5
            self.eval_Gt = eval_Gt_5
            self.eval_Vt = eval_Vt_5
        else:
            self.eval_gux = eval_gux
            self.eval_Gt = eval_Gt
            self.eval_Vt = eval_Vt

    def predict(self, u=None, sigma_u=None, g_extra_args=()):
        """Perform EKF prediction using *prior* state for Jacobians."""
        if len(g_extra_args) != 1:
            raise ValueError('Pass g_extra_args=(dt,) to predict().')
        dt = g_extra_args[0]
        prior = self.mu.copy()

        if self.dim_x == 3:
            if u is None:
                raise ValueError('Task 1 prediction requires u=[v,omega].')
            G = self.eval_Gt(*prior, *u, dt)
            V = self.eval_Vt(*prior, *u, dt)
        else:
            G = self.eval_Gt(*prior, dt)
            V = self.eval_Vt(*prior, dt)

        self.mu = self.eval_gux(prior, u, sigma_u, dt)
        self.Sigma = G @ self.Sigma @ G.T + V @ self.Mt @ V.T
        self.Sigma = (self.Sigma + self.Sigma.T) / 2.0

    def update(self, z, eval_hx, eval_Ht, Qt, Ht_args=(), hx_args=(),
               residual=residual_landmark):
        """Generic measurement update; select bearing wrapping per sensor."""
        H = eval_Ht(*Ht_args)
        SigmaHT = self.Sigma @ H.T
        self.S = H @ SigmaHT + Qt
        self.K = np.linalg.solve(self.S, SigmaHT.T).T
        z_hat = eval_hx(*hx_args)
        innovation = residual(z, z_hat)

        self.mu = self.mu + self.K @ innovation
        self.mu[2] = normalize_angle(self.mu[2])

        # Joseph covariance update is symmetric and positive-semidefinite.
        I_KH = self._I - self.K @ H
        self.Sigma = I_KH @ self.Sigma @ I_KH.T + self.K @ Qt @ self.K.T
        self.Sigma = (self.Sigma + self.Sigma.T) / 2.0
