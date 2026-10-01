# HMM(FORWARD)
import numpy as np
pi = np.array([0.5,0.3,0.2])
A = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.5, 0.3],
    [0.3, 0.2, 0.5]
])
B = np.array([
    [0.6, 0.3, 0.1],
    [0.1, 0.3, 0.6],
    [0.7, 0.2, 0.1]
])
O = [2,1,0,2]
T = len(O)
N = len(pi)
alpha = np.zeros((T,N))

for i in range(N):
  alpha[0][i] = pi[i]*B[i][O[0]]
for t in range(1,T):
  for j in range(N):
    alpha[t][j] = sum(
        alpha[t-1][i] * A[i][j]
        for i in range(N)
    )*B[j][O[t]]
likelihood = sum(alpha[T-1])
print("Forward Probability Table:")
print(alpha)
print("\nLikelihood = ",likelihood)