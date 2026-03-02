import numpy as np

def RealToFourier(u):
    """Given data u, at uniformly distributed grid-points
    0<=x<1,  calculates Fourier coefficients A[k] such that

    u = 0.5*A[0] + Sum_k(  A[k].real * cos(2 pi k x)
                         - A[k].imag * sin(2 pi k x)
                        )

    returns complex array A
    """
    A=np.fft.rfft(u)
    # adjust signs and magnitudes
    A *= (2./len(u))
    return A 

def FourierToReal(A):
    """Given spectral coefficients A[k], reconstruct the function
   0.5*A[0] + Sum_k(  A[k].real * cos(2 pi k x)
                    - A[k].imag * sin(2 pi k x)
                   )
    """
    u = np.fft.irfft(A)
    u = u * 0.5*len(u)
    return u

def dudx_Fourier(u):
    """Compute the first derivative dudx, using Foutier spectral methods.
    Assumes data is equally spaced in interval 0<=x<2pi and
    periodic."""
    A=RealToFourier(u)
    # multiply A[k] by  j k, where j is the imagimnary unit
    k= np.linspace(0, len(A)-1, len(A))
    A *= 1j*k
    # convert back tp real-space values
    dudx=FourierToReal(A)
    return dudx




def FourierHelpers_Complex(N, a=0, b=2*np.pi):
    """
    computes useful objects for coding of spectral methods for
    a real-valued Fourier-series truncated at order N/2:

        u(x) = Re (\sum_{n=0}^{N/2} c_n e^{inx})     (1)

    CONVENTION:
      storage of the coefficients is in one complex array of length N/2+1
         coefs = [a0, a_1-j*b_1, a_2-j*b_2, ... a_(N/2-1)-j*b_{N/2-1}, a_(N/2)]
      where a_n is the coefficient of cos(n*x), and b_n the coefficient of sin(n*x)
         
      That means coefs[0] and coefs[N/2] are real, thus we have the full N 
      degrees of freedom. 
"""
    if N%2!=0: raise "FourierHelpers works only for even N"

     #  X in [0, 2*pi]  ->   x in [a,b]:  x = A X + B 
    A=(b-a)/(2*np.pi)
    B=a
    def MappedCoords(X):
        return A*X + B

    M = int(N/2)  # biggest represented mode
    X = np.linspace(0., 2*np.pi, N, endpoint=False)
    x = MappedCoords(X)
    ToSpec = np.zeros( (M+1, N), dtype=complex)
    # cast to coefficients
    cbar=np.ones(M)
    cbar[0]=cbar[-1]=2.
    for j in range(0,M+1):  
        ToSpec[j,:]=2./N*np.exp(-1j * j * X)
    ToSpec[ 0,:] /= 2
    ToSpec[-1,:] /= 2

    ToPhys = np.zeros( (N,M+1), dtype=complex)
    # contains coefficients of u(x) = \sum_{n=0}^{N/2} a_n cos(nx) + \sum_{n=1}^{N/2-1} b_n sin(nx)
    for j in range(0,M+1):  
        ToPhys[:,j]=np.exp(1j * j * X)

    Dtilde = np.zeros( (M+1,M+1), dtype=complex )
    for k in range(0,M):
        Dtilde[k,   k ] =  1j*k
        # Dspec[M,M]=0, b /c cos(Mx)'=sin(Mx), which cannot be represented
    # so far Dtilde is in primitive coordinates (X), to get
    # d/dx = dX/dx d/dX,  must multiply by dX/dx = 1/A
    Dtilde *= 1./A

    # for grid-to-grid differentiation matrices 
    # only real part matters, owing to the Re im Eq. (1) above
    D1 = np.real(ToPhys @ Dtilde @ ToSpec)
    D2 = np.real(ToPhys @ Dtilde @ Dtilde @ ToSpec)

    return x, ToSpec, ToPhys, D1, D2




def FourierHelpers_Real(N, a=0, b=2*np.pi):
    """
    computes various real-space -> real-space transformation matrices
    for a Fourier basis, for real-valued functions.  In total we have 
    N degrees of freedom (about half of them being cosine and sine-modes).
       
    We define our spectral coefficients a_n and b_n such that 

        u(x) = \sum_{n=0}^{N/2} a_n cos(nx) + \sum_{n=1}^{N/2-1} b_n sin(nx)

    We require N to be even;  b_0 and b_N/2 do not exist, because the bass-function
    is zero at all collocation points.  

    CONVENTION:
      storage of the coefficients is in one real array of length N:
         coefs = [a0, ... a_(N/2), b1, ... b_(N/2-1)]
      This is a compact output format w/o any zeros, but makes for awkward indexing
      if one wants to find a specific mode. 

    RETURNS
      x  -- collocation points  np.array of length N covering [a,b]
            x[0] is at the lower bound a
      ToSpEC -- transformation matrix from real-space values (at xi) to spectral coefficients
      ToPhys -- transformation matrix from spectral coefficients to real space values (at xi)
      D1     -- transformation matrix of first derivative (real space -> real space)
      D2     -- transformation matrix of second derivative (real space -> real space)

      x is a np.array of length(N)
      all other quantities are NxN matrices
"""
    if N%2!=0: raise "FourierHelpers works only for even N"

    M = int(N/2)  # biggest represented mode

    #  X in [0, 2*pi]  ->   x in [a,b]:  x = A X + B 
    A=(b-a)/(2*np.pi)
    B=a
    def MappedCoords(X):
        return A*X + B

    X = np.linspace(0., 2*np.pi, N, endpoint=False)
    x=MappedCoords(X)
    A=np.zeros( (M+1, N))
    B=np.zeros( (M-1, N))
    A[0,:]=1/N   #  a_0 = 1/N sum_j u_j
    for j in range(1,M+1):
        A[j,:]=2/N*np.cos(j*X)      #  a_j = 2/N sum_k u_k cos(j*x_k)
    A[M,:] /= 2
    for j in range(1,M):
        B[j-1,:]=2/N*np.sin(j*X)  #  b_j = 2/N sum_k bu_k sin(j*x_k)
    
    # combine into one:
    ToSpec = np.zeros( (N,N) )
    ToSpec[0:M+1,:] = A
    ToSpec[M+1:N,:] = B


    ToPhys = np.zeros( (N,N) )
    # contains coefficients of u(x) = \sum_{n=0}^{N/2} a_n cos(nx) + \sum_{n=1}^{N/2-1} b_n sin(nx)
    for j in range(0,M+1):  # cos(N/2) is present
        ToPhys[:,j]=np.cos(j*X)
    for j in range(1,M): # sin(N/2 x) is _not_ present, so stop at M
        ToPhys[:,M+j]=np.sin(j*X)

    Dtilde = np.zeros( (N,N) )
    # a_0 -> 0, so don't touch Dspec[0,:]
    #   a_k cos(kx) + b_k sin(k x) ->   k*b_k cos(kx) - k*a_k sin(k x)
    for k in range(1,M):
        Dtilde[k,   M+k ] =  k
        Dtilde[M+k, k   ] = -k
    # highest a coefficient
    #   cos(N/2 x) -> -sin(N/2 x) ==>>  ZERO, because the sin(N/2 x) is not represented
    # this is accounted for because D[N/2,:] is never touched

    # so far Dtilde is in primitive coordinates (X), to get
    # d/dx = dX/dx d/dX,  must multiply by dX/dx = 1/A
    Dtilde *= 1./A

    D1 = ToPhys @ Dtilde @ ToSpec
    D2 = ToPhys @ Dtilde @ Dtilde @ ToSpec

    return x, ToSpec, ToPhys, D1, D2