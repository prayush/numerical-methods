import numpy as np


def ChebyshevInterpolate(u, ToSpec, x_out):
    """Interpolate the data u[i], assumed to be
    real-space values of a chebyshev expansion,
    to the grid-points in x_out.
    ToSpec is the transformation matrix from
    grid-point values to Chebyshev coefficients.
    """

    theta = np.arccos(x_out)  # avoid recomputation
    uspec = ToSpec @ u
    out = x_out * 0.0
    for k, coef in enumerate(uspec):
        out += coef * np.cos(k * theta)
    return out


def ChebyshevHelpers(N, a=-1.0, b=1.0):
    """ "return several useful objects for dealing with Chebyshev polynomials
    PARAMETERS:
      N - order of returned objects
      [a,b]  -- coordinate range to be considered.  I.e. the usual [-1,1] is linearly mapped to [a,b]

    RETURNS
      x  -- collocation points  np.array of length N covering [a,b]
            xi[0] is at the lower bound a
      ToSpEC -- transformation matrix from real-space values (at xi) to spectral coefficients
      ToPhys -- transformation matrix from spectral coefficients to real space values (at xi)
      D1     -- transformation matrix of first derivative (real space -> real space)
      D2     -- transformation matrix of second derivative (real space -> real space)
    """

    # linear mapping
    #  X in [1, 1]  ->   x in [a,b]:  x = A X + B
    A = 0.5 * (b - a)
    B = 0.5 * (a + b)

    def MappedCoords(X):
        return A * X + B

    # collocation points
    # Kidder(2000), Eq. (3.14)
    # '-' to make Xi[0] = lower boundary
    Xi = -np.cos(np.pi * np.arange(N) / (N - 1))
    xi = MappedCoords(Xi)

    # helper arrays
    # Kidder(2000), Eq. (3.13)
    c = np.ones(N)
    c[0] = 2.0
    # Kidder (2000). Eq. (3.16)
    cbar = np.ones(N)
    cbar[0] = 2.0
    cbar[N - 1] = 2.0

    # trafo matrix from spectral to grid-points
    # (this simply evaluates the Chebyshev polynomials
    #  T_k = cos(k*arccos(X)) at the collocation points X)
    ToPhys = np.ndarray((N, N))
    for k in range(N):
        ToPhys[:, k] = np.cos(k * np.arccos(Xi))

    # trafo matrix from grid-points to spectral coefficients
    #    rationale for (*) below:  if multiplied by another
    #    ToPhys a.k.a.  T_k(x_n), then the left-hand-side collapses
    #    to identity, and one has Eq. (3.15) from Kidder.
    ToSpec = np.ndarray((N, N))
    for k in range(N):
        ToSpec[k, :] = 2.0 / ((N - 1) * cbar[k] * cbar[:]) * ToPhys[:, k]  # (*)

    # print(ToSpec @ ToPhys)
    # print(ToPhys @ ToSpec)

    # spectral differentiaton matrix
    # Kidder (2000) Eq. (3.12) as a matrix
    Dtilde = np.ndarray((N, N))
    Dtilde.fill(0.0)
    # I am sure there's a better way
    for k in reversed(range(N - 1)):
        if k <= N - 3:
            Dtilde[k, :] = 1 / c[k] * Dtilde[k + 2, :]
        Dtilde[k, k + 1] += 2 * (k + 1) / c[k]
    # multiply by -1 to account for our choice that the first collocation
    # point is at the __lower__ boundary
    # so far Dtilde is in primitive coordinates (X), to get
    # d/dx = dX/dx d/dX,  must multiply by dX/dx = 1/A
    Dtilde *= 1.0 / A

    # full first derivative physical-to-physical differentiaton matrix
    D1 = ToPhys @ Dtilde @ ToSpec

    # full second derivative physical-to-physical differentiaton matrix
    D2 = ToPhys @ Dtilde @ Dtilde @ ToSpec

    return xi, ToSpec, ToPhys, D1, D2
