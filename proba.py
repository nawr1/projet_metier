import numpy as np
import matplotlib.pyplot as plt
A = np.array([[0, 0.7, 0.3],
              [0.3, 0.4, 0.3],
              [0.5, 0.5, 0]])

def PremTermesLoiXn(x0, A, n):
    V = np.zeros((n+1, 3))
    M = np.eye(3)
    
    for k in range(n+1):
        V[k] = M[x0-1]   # correction ici
        M = np.dot(M, A)
    
    return V


# paramètres
n = 50
x0 = 1

Y = range(n+1)
V = PremTermesLoiXn(x0, A, n)

plt.plot(Y, V[:,0], '+', label='P(X_n=1)')
plt.plot(Y, V[:,1], '.', label='P(X_n=2)')
plt.plot(Y, V[:,2], 'x', label='P(X_n=3)')
plt.legend(loc='best')
plt.show()