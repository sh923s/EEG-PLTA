import numpy as np
from scipy.linalg import fractional_matrix_power

class online_EA:
    def __init__(self):
        self.num_samples = 0
        self.R = 0
    
    def align(self, x):
        cov = np.cov(x)
        self.R = (self.num_samples * self.R + cov) / (self.num_samples + 1)
        self.num_samples += 1
        sqrtRefEA = fractional_matrix_power(self.R, -0.5)
        return np.dot(sqrtRefEA, x)
    
    def reset(self):
        self.num_samples = 0
        self.R = 0
        
        