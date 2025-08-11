import numpy as np

class LinUCB:
    def __init__(self, n_arms: int, d: int, alpha: float=0.8):
        self.n = n_arms
        self.d = d
        self.alpha = alpha
        self.A = [np.eye(d) for _ in range(n_arms)]   # d x d
        self.b = [np.zeros((d, 1)) for _ in range(n_arms)]

    def select(self, x: np.ndarray) -> int:
        x = x.reshape(-1,1)
        ucb_vals = []
        for a in range(self.n):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            mu = float(theta.T @ x)
            bonus = self.alpha * float(np.sqrt(x.T @ A_inv @ x))
            ucb_vals.append(mu + bonus)
        return int(np.argmax(ucb_vals))

    def update(self, arm: int, x: np.ndarray, reward: float):
        x = x.reshape(-1,1)
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x
