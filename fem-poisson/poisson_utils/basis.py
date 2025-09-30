import numpy as np
from scipy.special import gamma

def polynomial(x, alpha, beta, N):
    """
    Evaluate Jacobi Polynomial of type (alpha,beta) > -1
    (alpha+beta != -1) at points x for order N and returns P with shape (len(x),)
    Note: They are normalized to be orthonormal.
    """
    xp = np.atleast_1d(x).astype(float)
    dims = xp.shape
    if xp.ndim == 1:
        xp = xp.reshape(1, -1)  # make row vector
    elif xp.shape[0] > 1 and xp.shape[1] == 1:
        xp = xp.T

    PL = np.zeros((N+1, xp.shape[1]))

    # Initial values P_0(x) and P_1(x)
    gamma0 = 2**(alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+1)
    PL[0, :] = 1.0/np.sqrt(gamma0)
    if N == 0:
        return PL[0, :].T

    gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0
    PL[1, :] = ((alpha+beta+2)*xp/2 + (alpha-beta)/2)/np.sqrt(gamma1)
    if N == 1:
        return PL[1, :].T

    aold = 2/(2+alpha+beta)*np.sqrt((alpha+1)*(beta+1)/(alpha+beta+3))

    # Forward recurrence
    for i in range(1, N):
        h1 = 2*i + alpha + beta
        anew = 2/(h1+2)*np.sqrt((i+1)*(i+1+alpha+beta)*(i+1+alpha)*(i+1+beta)/(h1+1)/(h1+3))
        bnew = - (alpha**2 - beta**2) / (h1*(h1+2))
        PL[i+1,:] = ( -aold*PL[i-1,:] + (xp-bnew)*PL[i,:] ) / anew
        aold = anew

    return PL[N, :].T

def gradient(r, alpha, beta, N):
    """
    Evaluate the derivative of the Jacobi polynomial of type (alpha,beta)>-1,
    at points r for order N and returns dP with shape (len(r),)
    """
    r = np.atleast_1d(r)
    dP = np.zeros(r.shape)
    if N == 0:
        dP[:] = 0.0
    else:
        dP = np.sqrt(N*(N+alpha+beta+1)) * polynomial(r, alpha+1, beta+1, N-1)
    return dP

def gauss(alpha, beta, N):
    """
    Compute the N'th order Gauss quadrature points, x,
    and weights, w, associated with the Jacobi polynomial, of type (alpha,beta) > -1 ( <> -0.5).
    """
    if N == 0:
        x = np.array([(alpha-beta)/(alpha+beta+2)], dtype=float)
        w = np.array([2.0], dtype=float)
        return x, w

    zero_to_N = np.arange(N+1)
    one_to_N = zero_to_N[1:]

    h1 = 2*zero_to_N + alpha + beta

    J = np.diag(-0.5*(alpha**2 - beta**2) / (h1+2) / h1) + \
        np.diag(2/(h1[:-1]+2)*np.sqrt(
                    one_to_N * (one_to_N+alpha+beta) * (one_to_N+alpha) * \
                    (one_to_N+beta) / (h1[:-1]+1) / (h1[:-1]+3)
                    )
                , 1)


    if (alpha+beta) < 10*np.finfo(float).eps:
        J[0,0] = 0.0

    J = J + J.T
    # Compute quadrature by eigenvalue solve
    D, V = np.linalg.eigh(J)
    x = D
    w = (V[0,:])**2 * 2**(alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+1)
    return x, w

def gll(alpha, beta, N):
    """
    Compute the N'th order Gauss Lobatto quadrature points, x,
    and weights, w, associated with the Jacobi polynomial, of type (alpha,beta) > -1 ( <> -0.5).
    """
    x = np.zeros(N+1)
    w = np.zeros(N+1)
    if N == 1:
        x[0] = -1.0
        x[1] = 1.0
        w[0] = 1.0
        w[1] = 1.0
        return x, w

    # Get interior Gauss points
    xint, _ = gauss(alpha+1, beta+1, N-2)
    x = np.concatenate(([-1.0], xint, [1.0]))

    # compute the weights
    P = polynomial(x, alpha, beta, N)
    adgammaN = (2.0*N + alpha + beta + 1.0) / (N * (alpha + beta + N + 1.0))

    w = adgammaN / (P*P)
    w[0]   = w[0]   * (1.0 + alpha)
    w[-1]  = w[-1]  * (1.0 + beta)
    return x, w


def test():
    x, w = gll(0.2, 0.2, 8)
    print(x)
    print(w)

if __name__ == "__main__":
    test()