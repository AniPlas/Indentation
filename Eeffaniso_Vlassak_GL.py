# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:42:34 2023

@author: richeton5
"""

# -*- coding: utf-8 -*-
import numpy as np
import time

st = time.time()

#    
Nint = 30 # Number of points to compute the Gauss-Legendre quadrature

# Elastic constants of Ni in GPa 
c11=244          
c12=158
c44=102

# Euler angles in degrees
phi1 = 271;
phi = 147;
phi2 = 242;

# Euler angles in radians
phi1 = phi1*np.pi/180
phi  = phi*np.pi/180
phi2 = phi2*np.pi/180

def mrot(X,theta):
    R = np.zeros((3,3))
    s = theta
    x = X[0]
    y = X[1]
    z = X[2]
    ##%%%%%%%%%%%%%%%%%%%%% rotation clockwise %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Euler angles given by EBSD : passing from global frame to crystal frame 
    # by anti-clockwise rotations [Bunge convention]
    #  --> to determine the passing matrix global --> crystal, we write the
    #  rotation matrices in the clockwise sense
    R[0,0] = np.cos(s)+(x**2)*(1-np.cos(s))
    R[0,1] = x*y*(1-np.cos(s))+z*np.sin(s)
    R[0,2] = x*z*(1-np.cos(s))-y*np.sin(s)
    R[1,0] = y*x*(1-np.cos(s))-z*np.sin(s)
    R[1,1] = np.cos(s)+(y**2)*(1-np.cos(s))
    R[1,2] = y*z*(1-np.cos(s))+x*np.sin(s)
    R[2,0] = z*x*(1-np.cos(s))+y*np.sin(s)
    R[2,1] = z*y*(1-np.cos(s))-x*np.sin(s)
    R[2,2] = np.cos(s)+(z**2)*(1-np.cos(s))
    return R

def cubicelasticconst(c11,c12,c44,meth=None,pert=None):
    # generate elastic stiffness tensor C_ijkl
    # C_ijkl is a rank 4 tensor
    # pert: perturbation to c44 when the medium is isotropic
    # ==============================================
    # perturbation for isotropic medium
    tol = 1e-6 #tolerance
    
    if meth == None and pert == None:
        meth = 'm'
        pert = 1e-6#;  default perturbation rate if the media is isotropic
    elif meth == 'm':
        pert = 1e-6
    # perturb c44 specifically for matrix formalism
    if (meth == 'm') and (abs(2*c44+c12-c11) < tol):
        c44 = c44 + pert    
    # ==============================================
    # construct C_ijkl
    # initializing C
    C = np.zeros((3,3,3,3))
    # construct delta function
    delta = np.eye(3)
    # Define H
    H = 2*c44+c12-c11;
    # For cubic crystals
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i,j,k,l] = c44*(delta[i,k]*delta[j,l] + delta[i,l]*delta[j,k]) + \
                                 c12*delta[i,j]*delta[k,l] - \
                                 H*delta[i,j]*delta[k,l]*delta[i,k]
    
    return C

def legzo(n, a=-1, b=1):
    """
    Compute the zeros of Legendre polynomial Pn(x) in the interval [a,b],
    and the corresponding weighting coefficients for Gauss-Legendre integration
    
    Parameters
    ----------
    n : int
        Order of the Legendre polynomial
    a : float, optional
        Lower boundary (default=-1)
    b : float, optional
        Upper boundary (default=1)
    
    Returns
    -------
    x : ndarray of floats
        Zeros of the Legendre polynomial
    w : ndarray of floats
        Corresponding weighting coefficients
    """
    eps = np.finfo(float).eps
    x = np.zeros(n)
    w = np.zeros(n)
    m = int((n+1)/2)
    h = b-a

    for ii in range(1, m+1):
        z = np.cos(np.pi*(ii-0.25)/(n+0.5)) # Initial estimate.
        z1 = z+1
        while abs(z-z1) > eps:
            p1 = 1
            p2 = 0
            for jj in range(1, n+1):
                p3 = p2
                p2 = p1
                p1 = ((2*jj-1)*z*p2-(jj-1)*p3)/jj # The Legendre polynomial.
            pp = n*(z*p1-p2)/(z**2-1) # The L.P. derivative.
            z1 = z
            z = z1-p1/pp
        x[ii-1] = z # Build up the abscissas.
        x[n-ii] = -z
        w[ii-1] = h/((1-z**2)*(pp**2)) # Build up the weights.
        w[n-ii] = w[ii-1]

    if a != -1 or b != 1:
        x = (x+1)*(h/2) + a
    
    return x, w

###############################################################################
x = [1,0,0]
y = [0,1,0]
z = [0,0,1]

# Gauss points and weights for the integration
omegaa, w = legzo(Nint,0,2*np.pi)
# print(omegaa,w)

# Macroscopic direction along z
ma = np.copy(z)
# A direction normal to ma
q = np.copy(y)

#  Cubic stiffness matrix
C = cubicelasticconst(c11,c12,c44)


# Rotation matrix
R1 = mrot(z,phi1)
R2 = mrot(x,phi)
R3 = mrot(z,phi2)
RR = R3.dot(R2).dot(R1)

# Crystal orientation
n = np.matmul(RR,ma) # normal to indentation surface direction
T = np.matmul(RR,q) # a direction normal to n

n = n/np.linalg.norm(n,2) #norm(n);
T = T/np.linalg.norm(T,2) #norm(T);

#  Initialization 
somme2 = np.zeros((Nint))
Estar = 0
for s in range(1, Nint+1):
#  Initializing nn nm mn mm and nn_1,2,3
#  Each m and n are angle dependent
    nn = np.zeros((3,3))
    nm = np.zeros((3,3))
    mn = np.zeros((3,3))
    mm = np.zeros((3,3))
          
    omega = omegaa[s-1]
    R = np.cross(n,T)
    R = R/np.linalg.norm(R,2) #norm(R);
    t = T*np.cos(omega)+R*np.sin(omega)
    m = np.cross(n,t)
    m = m/np.linalg.norm(m,2) #norm(m);
    
#  Defining nn mm nm mn for each theta
    for j in range(3):
        for k in range(3):
            for i in range(3):
                for l in range(3):
                    nn[j,k] = nn[j,k] + n[i]*C[i,j,k,l]*n[l]
                    nm[j,k] = nm[j,k] + n[i]*C[i,j,k,l]*m[l]
                    mm[j,k] = mm[j,k] + m[i]*C[i,j,k,l]*m[l]
                    
    mn = nm.T #nm'
    
#  Create 6x6 matrix N0
    invnn = np.linalg.inv(nn)
    t1 = -np.hstack((np.matmul(invnn,nm), invnn)) 
    t2 = -np.hstack((mn.dot(invnn).dot(nm)-mm, np.matmul(mn,invnn))) 
    N0 = np.vstack((t1,t2))
#  Solve Eigen equation
    D, V = np.linalg.eig(N0)
    D = np.eye(6) * D
#  Initializing A, L and P
    A = np.zeros((3,3),dtype=complex)
    L = np.zeros((3,3),dtype=complex)
    P = np.zeros((3),dtype=complex)
#  Define A, L and P
    for alpha in range(3): 
        if D[(2*alpha),(2*alpha)].imag > 0:
            P[alpha] = D[(2*alpha),(2*alpha)]
            A[:,alpha] = V[0:3,2*alpha]
            L[:,alpha] = V[3:6,2*alpha]
        else:
            P[alpha] = D[2*alpha-1,2*alpha-1]
            A[:,alpha] = V[0:3,2*alpha-1]
            L[:,alpha] = V[3:6,2*alpha-1] 
#  Normalize A and L
    for alpha in range(3): 
#  AialphaLibeta+AibetaLialpha=delta(alpha,beta)
        ralpha = np.sqrt(2*np.sum(A[:,alpha]*L[:,alpha]))
        A[:,alpha] = A[:,alpha]/ralpha
        L[:,alpha] = L[:,alpha]/ralpha   

#  Construct Q, B, S matrices using A and L
    I = 0.0 + 1j 
    B = 2*I*(np.matmul(L,L.T))
    invB = np.linalg.inv(B.real)
    
    somme2 = 0.
    for i in range(3): 
         for j in range(3): 
             somme2 = somme2 + n[i]*invB[j,i]*n[j]
         
    Estar = Estar + somme2*w[s-1]

#  Indentation modulus
Estar = 4*np.pi/Estar
print("Indentation modulus is:  " +str(Estar))

# get the end time
et = time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
