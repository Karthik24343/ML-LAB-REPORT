import numpy as np

# Initial probabilities
pi = np.array([0.5, 0.3, 0.2])

# Transition matrix
A = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.5, 0.3],
    [0.3, 0.2, 0.5]
])

# Emission matrix (example)
B = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.5, 0.3],
    [0.3, 0.2, 0.5]
])

# Observed sequence (0-based symbols)
obs = [2, 0, 2]

T = len(obs)
N = len(pi)

# Initialize backward matrix
beta = np.zeros((T, N))
beta[T-1] = 1

# Backward algorithm
for t in range(T-2, -1, -1):
    for i in range(N):
        for j in range(N):
            beta[t, i] += A[i, j] * B[j, obs[t+1]] * beta[t+1, j]

print("Backward probabilities:")
print(beta)

# Probability of the observed sequence
P = np.sum(pi * B[:, obs[0]] * beta[0])
print("Probability:", P)