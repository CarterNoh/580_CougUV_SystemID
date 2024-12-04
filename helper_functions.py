import numpy as np
from numba import njit

@njit
def Smtrx(a):
        """
        S = Smtrx(a) computes the 3x3 vector skew-symmetric matrix S(a) = -S(a)'.
        The cross product satisfies: a x b = S(a)b. 
        """
    
        S = np.array([ 
            [ 0, -a[2], a[1] ],
            [ a[2],   0,     -a[0] ],
            [-a[1],   a[0],   0 ]  ])

        return S

@njit
def Hmtrx(r):
        """
        H = Hmtrx(r) computes the 6x6 system transformation matrix
        H = [eye(3)     S'
            zeros(3,3) eye(3) ]       Property: inv(H(r)) = H(-r)

        If r = r_bg is the vector from the CO to the CG, the model matrices in CO and
        CG are related by: M_CO = H(r_bg)' * M_CG * H(r_bg). Generalized position and
        force satisfy: eta_CO = H(r_bg)' * eta_CG and tau_CO = H(r_bg)' * tau_CG 
        """

        H = np.identity(6,float)
        H[0:3, 3:6] = Smtrx(r).T

        return H

@njit
def m2c(M, nu):
        """
        C = m2c(M,nu) computes the Coriolis and centripetal matrix C from the
        mass matrix M and generalized velocity vector nu (Fossen 2021, Ch. 3)
        """

        M = 0.5 * (M + M.T)     # systematization of the inertia matrix

        if (len(nu) == 6):      #  6-DOF model
        
            M11 = M[0:3,0:3]
            M12 = M[0:3,3:6] 
            M21 = M12.T
            M22 = M[3:6,3:6] 
        
            nu1 = nu[0:3]
            nu2 = nu[3:6]
            dt_dnu1 = np.dot(M11,nu1) + np.dot(M12,nu2)
            dt_dnu2 = np.dot(M21,nu1) + np.dot(M22,nu2)

            #C  = [  zeros(3,3)      -Smtrx(dt_dnu1)
            #      -Smtrx(dt_dnu1)  -Smtrx(dt_dnu2) ]
            C = np.zeros( (6,6) )    
            C[0:3,3:6] = -Smtrx(dt_dnu1)
            C[3:6,0:3] = -Smtrx(dt_dnu1)
            C[3:6,3:6] = -Smtrx(dt_dnu2)
                
        else:   # 3-DOF model (surge, sway and yaw)
            #C = [ 0             0            -M(2,2)*nu(2)-M(2,3)*nu(3)
            #      0             0             M(1,1)*nu(1)
            #      M(2,2)*nu(2)+M(2,3)*nu(3)  -M(1,1)*nu(1)          0  ]    
            C = np.zeros( (3,3) ) 
            C[0,2] = -M[1,1] * nu[1] - M[1,2] * nu[2]
            C[1,2] =  M[0,0] * nu[0] 
            C[2,0] = -C[0,2]       
            C[2,1] = -C[1,2]
            
        return C

@njit
def ssa(angle):
    """
    angle = ssa(angle) returns the smallest-signed angle in [ -pi, pi )
    """
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
        
    return angle 

@njit
def sat(x, x_min, x_max):
    """
    x = sat(x,x_min,x_max) saturates a signal x such that x_min <= x <= x_max
    """
    if x > x_max:
        x = x_max 
    elif x < x_min:
        x = x_min
        
    return x 

@njit
def Rzyx(phi, theta, psi):
        """
        R = Rzyx(phi,theta,psi) computes the Euler angle rotation matrix R in SO(3)
        using the zyx convention
        converts body to world 
        """
        
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cth  = np.cos(theta)
        sth  = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        
        R = np.array([
            [ cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth ],
            [ spsi*cth,  cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi ],
            [ -sth,      cth*sphi,                 cth*cphi ] ])

        return R

def Tzyx(phi, theta):
    """
    T = Tzyx(phi,theta) computes the Euler angle attitude
    transformation matrix T using the zyx convention
    body frame angular velocities (p,q,r) to Euler angle rates (phi_dot, theta_dot, psi_dot)
    """
    
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth  = np.cos(theta)
    sth  = np.sin(theta)    

    try: 
        T = np.array([
            [ 1,  sphi*sth/cth,  cphi*sth/cth ],
            [ 0,  cphi,          -sphi],
            [ 0,  sphi/cth,      cphi/cth] ])
        
    except ZeroDivisionError:  
        print ("Tzyx is singular for theta = +-90 degrees." )
        
    return T

@njit
def Hoerner(B,T):
    """
    CY_2D = Hoerner(B,T)
    Hoerner computes the 2D Hoerner cross-flow form coeff. as a function of beam 
    B and draft T.The data is digitized and interpolation is used to compute 
    other data point than those in the table
    """
    
    # DATA = [B/2T  C_D]
    DATA1 = np.array([
        0.0109,0.1766,0.3530,0.4519,0.4728,0.4929,0.4933,0.5585,0.6464,0.8336,
        0.9880,1.3081,1.6392,1.8600,2.3129,2.6000,3.0088,3.4508, 3.7379,4.0031 
        ])
    DATA2 = np.array([
        1.9661,1.9657,1.8976,1.7872,1.5837,1.2786,1.2108,1.0836,0.9986,0.8796,
        0.8284,0.7599,0.6914,0.6571,0.6307,0.5962,0.5868,0.5859,0.5599,0.5593 
        ])

    CY_2D = np.interp( B / (2 * T), DATA1, DATA2 )
        
    return CY_2D








