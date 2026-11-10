"""Kinematic models for the two agents.

Coordinates are the BEV map frame: x = right (m), z = forward (m).
Heading psi is measured from +z toward +x (psi = 0 -> straight ahead).
The agent's reference point starts at the camera position (x=0, z=0).

All rollouts are vectorized: N candidate control sequences x T steps.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class CarAgent:
    """Kinematic bicycle model (reference point = rear axle)."""
    name: str = "car"
    wheelbase: float = 2.71        # KITTI VW Passat
    length: float = 4.5
    width: float = 1.8
    height: float = 1.6
    safety_margin: float = 0.25
    max_steer: float = 0.55        # rad
    max_steer_rate: float = 0.5    # rad/s
    max_accel: float = 2.0         # m/s^2
    max_decel: float = 6.0         # m/s^2 (emergency)
    max_speed: float = 15.0        # m/s
    max_lat_accel: float = 3.0     # m/s^2 comfort limit
    min_obstacle_height: float = 0.3
    horizon: float = 4.0           # s
    dt: float = 0.1
    goal_lookahead: float = 40.0   # m, how far ahead to look for a goal
    start_z: float = -1.0          # rear axle sits ~1 m behind the roof camera
    mode_temperature: float = 5.0  # softmax temperature (cost units ~ metres) for mode probs

    def footprint(self):
        """Covering circles (offsets along heading from the rear axle, radius)."""
        n = 5
        r = np.hypot(self.length / (2 * n), self.width / 2) + self.safety_margin
        rear_overhang = 1.0
        centers = -rear_overhang + self.length / (2 * n) * (2 * np.arange(n) + 1)
        return centers, r

    def sample_controls(self, v0):
        """Two-segment steering profiles x constant accelerations -> (N, T, 2) [accel, steer]."""
        T = int(round(self.horizon / self.dt))
        half = T // 2
        # quadratic spacing: dense near straight, where high-speed driving happens
        s = np.linspace(-1.0, 1.0, 17)
        steer = self.max_steer * np.sign(s) * s ** 2
        accels = np.array([-self.max_decel, -3.0, -1.0, 0.0, 1.0, self.max_accel])
        s1, s2, ac = np.meshgrid(steer, steer, accels, indexing="ij")
        s1, s2, ac = s1.ravel(), s2.ravel(), ac.ravel()
        u = np.zeros((len(s1), T, 2))
        u[:, :, 0] = ac[:, None]
        u[:, :half, 1] = s1[:, None]
        u[:, half:, 1] = s2[:, None]
        return self._limit_steer_rate(u)

    def _limit_steer_rate(self, u):
        steer = u[:, :, 1].copy()
        d_max = self.max_steer_rate * self.dt
        prev = np.zeros(len(steer))  # assume wheels straight at t=0
        for t in range(steer.shape[1]):
            steer[:, t] = prev + np.clip(steer[:, t] - prev, -d_max, d_max)
            prev = steer[:, t]
        u[:, :, 1] = steer
        return u

    def rollout(self, state0, u):
        """state0 = (x, z, psi, v); u: (N, T, 2). Returns (N, T+1, 4)."""
        N, T, _ = u.shape
        s = np.zeros((N, T + 1, 4))
        s[:, 0] = state0
        for t in range(T):
            x, z, psi, v = s[:, t].T
            a, d = u[:, t].T
            s[:, t + 1, 0] = x + v * np.sin(psi) * self.dt
            s[:, t + 1, 1] = z + v * np.cos(psi) * self.dt
            s[:, t + 1, 2] = psi + v / self.wheelbase * np.tan(d) * self.dt
            s[:, t + 1, 3] = np.clip(v + a * self.dt, 0.0, self.max_speed)
        return s

    def lateral_accel(self, states, u):
        v = states[:, 1:, 3]
        return v ** 2 / self.wheelbase * np.abs(np.tan(u[:, :, 1]))


@dataclass
class RobotAgent:
    """Differential-drive / unicycle model."""
    name: str = "robot"
    radius: float = 0.3
    height: float = 0.6
    safety_margin: float = 0.1
    max_speed: float = 1.5         # m/s
    max_accel: float = 1.0
    max_decel: float = 2.0
    max_yaw_rate: float = 1.5      # rad/s
    max_lat_accel: float = 1.5
    min_obstacle_height: float = 0.12
    horizon: float = 5.0
    dt: float = 0.1
    goal_lookahead: float = 12.0
    start_z: float = 0.0           # sensor at robot centre
    mode_temperature: float = 1.0

    def footprint(self):
        return np.array([0.0]), self.radius + self.safety_margin

    def sample_controls(self, v0):
        """(N, T, 2) [accel, yaw_rate] with two-segment yaw-rate profiles."""
        T = int(round(self.horizon / self.dt))
        half = T // 2
        w = np.linspace(-self.max_yaw_rate, self.max_yaw_rate, 15)
        accels = np.array([-self.max_decel, -0.5, 0.0, 0.5, self.max_accel])
        w1, w2, ac = np.meshgrid(w, w, accels, indexing="ij")
        w1, w2, ac = w1.ravel(), w2.ravel(), ac.ravel()
        u = np.zeros((len(w1), T, 2))
        u[:, :, 0] = ac[:, None]
        u[:, :half, 1] = w1[:, None]
        u[:, half:, 1] = w2[:, None]
        return u

    def rollout(self, state0, u):
        N, T, _ = u.shape
        s = np.zeros((N, T + 1, 4))
        s[:, 0] = state0
        for t in range(T):
            x, z, psi, v = s[:, t].T
            a, w = u[:, t].T
            s[:, t + 1, 0] = x + v * np.sin(psi) * self.dt
            s[:, t + 1, 1] = z + v * np.cos(psi) * self.dt
            s[:, t + 1, 2] = psi + w * self.dt
            s[:, t + 1, 3] = np.clip(v + a * self.dt, 0.0, self.max_speed)
        return s

    def lateral_accel(self, states, u):
        return np.abs(states[:, 1:, 3] * u[:, :, 1])


AGENTS = {"car": CarAgent, "robot": RobotAgent}
