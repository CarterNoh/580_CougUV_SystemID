import numpy as np

class Coug:
    '''
    This class sets up the vehicle for the CS 580 class project on system ID. 
    '''

    def __init__(self, params=None):

        ################# INITIALIZE ALL PARAMETERS #######################
        
        # Initilize state and control vectors
        self.nu = np.array([0, 0, 0, 0, 0, 0], float) # velocity vector in body frame
        self.eta = np.array([0, 0, 0, 0, 0, 0], float) # velocity vector in NED frame
        self.u_actual = np.array([0, 0, 0, 0], float)    # control input vector

        # Environment Constants
        # TODO: should some of these be class variable's instead?
        self.D2R = np.pi / 180          # deg2rad
        self.rho = 1000                 # density of water (kg/m^3)
        self.g = 9.81                   # acceleration of gravity (m/s^2)
        self.V_c = 0                    # velocity of current (m/s)
        self.beta_c = 0 * self.D2R      # current angle (rad)
        
        # Vehical physical Parameters
        self.r_bg = np.array([0, 0, 0.02], float)    # CG w.r.t. to the CO
        self.r_bb = np.array([0, 0, 0], float)       # CB w.r.t. to the CO
        self.m = 16
        self.L = 1.6                    # length (m)
        self.diam = 0.19                # cylinder diameter (m)
        self.area_fraction = np.pi/4    # relates vehicle effective area to length and width. pi/4 for a spheroid

        # Low-speed linear damping matrix parameters
        self.T_surge = 20               # time constant in surge (s)
        self.T_sway = 20                # time constant in sway (s)
        self.T_heave = self.T_sway      # equal for for a cylinder-shaped AUV
        self.zeta_roll = 0.3            # relative damping ratio in roll
        self.zeta_pitch = 0.8           # relative damping ratio in pitch
        self.T_yaw = 1                  # time constant in yaw (s)

        # Other damping/force parameters
        self.Cd = 0.42                  # Coefficient of drag for entire vehicle
        self.r44 = 0.3                  # Added #TODO: what is this?

        # Fin Parameters
        self.S_fin = 0.00697            # Surface area of one side of a fin
        self.x_fin = -self.L/2          # X distance from center of mass to fins
        self.fin_center = 0.07          # Positive Z distance from center of mass to center of pressure
        self.CL_delta_r = 0.5           # rudder lift coefficient
        self.CL_delta_s = 0.7           # stern-plane lift coefficient
        self.deltaMax_r = 20 * self.D2R # max rudder angle (rad)
        self.deltaMax_s = 20 * self.D2R # max stern plane angle (rad)
        self.T_delta = 0.1              # rudder/stern plane time constant (s)

        # Propellor parameters
        self.D_prop = 0.14              # propeller diameter (m)
        self.t_prop = 0.1               # thrust deduction number
        self.Ja_max = 0.6632            # 
        self.Va = 0.944                 # advance speed constant
        self.KT_0 = 0.4566              # 
        self.KQ_0 = 0.0700              # 
        self.KT_max = 0.1798            # 
        self.KQ_max = 0.0312            # 
        self.nMax = 2000                # max propeller speed (rpm)
        self.T_n = 0.1                  # propeller time constant (s)


        ################### OVERWRITE DESIRED PARAMETERS ######################

        if params is not None:
            self.override_params(params)

        self.calc_parameters()
        
    def override_params(self, params):
        '''
        Input: 
        - params: a dictionary of parameters we want to make overwriteable.
         Keys: 
        '''
        for key in params:
            if key not in self.__dict__:
                raise ValueError(f"Invalid parameter: {key}")
            else:
                self.__dict__[key] = params[key]
    
    def calc_parameters(self):
        '''Calculate additional vehicle parameters based on inputs.'''

        ### Vehicle Geometry ###
        a = self.L/2
        b = self.diam/2                          

        ###### MASS MATRIX #######
        ### Rigid-body mass matrix expressed in CO ###
        # Assumes a spheroid body
        # m = 4/3 * np.pi * self.rho * a * b**2       # mass of spheriod 
        Ix = (2/5) * self.m * b**2                       # moment of inertia
        Iy = (1/5) * self.m * (a**2 + b**2)
        Iz = Iy
        MRB_CG = np.diag([self.m, self.m, self.m, Ix, Iy, Iz ])   # MRB expressed in the CG     
        H_rg = self.Hmtrx(self.r_bg)
        self.MRB = H_rg.T @ MRB_CG @ H_rg           # MRB expressed in the CO

        ### Added Mass Matrix
        MA_44 = self.r44 * Ix # added moment of inertia in roll: A44 = r44 * Ix
        # Lamb's k-factors
        e = np.sqrt( 1-(b/a)**2 )
        alpha_0 = ( 2 * (1-e**2)/pow(e,3) ) * ( 0.5 * np.log( (1+e)/(1-e) ) - e )  
        beta_0  = 1/(e**2) - (1-e**2) / (2*pow(e,3)) * np.log( (1+e)/(1-e) )
        k1 = alpha_0 / (2 - alpha_0)
        k2 = beta_0  / (2 - beta_0)
        k_prime = pow(e,4) * (beta_0-alpha_0) / ( 
            (2-e**2) * ( 2*e**2 - (2-e**2) * (beta_0-alpha_0) ) )   
        self.MA = np.diag([self.m*k1, self.m*k2, self.m*k2, MA_44, k_prime*Iy, k_prime*Iy ])
          
        ### Mass matrix including added mass
        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)


        ##### GRAVITY VECTOR ######
        self.W = self.m * self.g
        self.B = self.W

        ##### HYDROYNAMICS #####
        #  CD_0, i.e. zero lift and alpha = 0
        self.S = self.area_fraction * self.L * self.diam  # effective surface area for drag
        self.CD_0 = self.Cd * np.pi * b**2 / self.S       # Parasitic drag coefficient

        # Natural frequencies in roll and pitch
        self.w_roll =  np.sqrt(self.W * (self.r_bg[2]-self.r_bb[2]) / self.M[3][3])
        self.w_pitch = np.sqrt(self.W * (self.r_bg[2]-self.r_bb[2]) / self.M[4][4])
            
     
        ##### CONTROL SURFACES #####
        # Tail rudder parameters
        self.A_fin = self.S_fin        # rudder area (m2)
        self.x_r = -a               # rudder x-position (m)

        # Stern-plane parameters (double)
        self.A_s = 2 * self.S_fin        # stern-plane area (m2)
        self.x_s = -a               # stern-plane z-position (m)

    def dynamics(self, u_control, sampleTime):
        """
        Integrates the AUV equations of motion using Euler's method.

        :param array-like eta: State/pose of the vehicle in the world frame.
        :param array-like nu: Velocity of the vehicle in the body frame.
        :param array-like nu_dot: Acceleration of the vehicle in the body frame.
        :param array-like u_actual: Current control surface position.
        :param array-like u_control: Commanded control surface position.
        :param float sampleTime: Time since the last step.

        :returns: Three arrays: nu, u_actual, and nu_dot.
        """

        ### Velocity of Vehicle and Current ###
        u_c = self.V_c * np.cos(self.beta_c - self.eta[5])  # current surge velocity
        v_c = self.V_c * np.sin(self.beta_c - self.eta[5])  # current sway velocity

        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float) # current velocity 
        Dnu_c = np.array([self.nu[5]*v_c, -self.nu[5]*u_c, 0, 0, 0, 0],float) # derivative
        nu_r = self.nu - nu_c                               # relative velocity between vehicle and current      
        alpha = np.arctan2(nu_r[2], nu_r[0])         # angle of attack 
        U = np.sqrt(self.nu[0]**2 + self.nu[1]**2 + self.nu[2]**2)  # vehicle speed
        U_r = np.sqrt(nu_r[0]**2 + nu_r[1]**2 + nu_r[2]**2)  # relative speed

        ### Coriolis Matrix ###
        # Rigid-body/added mass Coriolis/centripetal matrices expressed in the CO
        CRB = self.m2c(self.MRB, nu_r)
        CA  = self.m2c(self.MA, nu_r)
        # CA-terms in roll, pitch and yaw can destabilize the model if quadratic
        # rotational damping is missing. These terms are assumed to be zero. 
        CA[4][0] = 0     # Quadratic velocity terms due to pitching
        CA[0][4] = 0  
        CA[2][4] = 0
        CA[5][0] = 0     # Munk moment in yaw
        CA[5][1] = 0
        CA[1][5] = 0
        C = CRB + CA

        ### Dissipative Matrix ###
        D = np.diag([
            self.M[0][0] / self.T_surge,
            self.M[1][1] / self.T_sway,
            self.M[2][2] / self.T_heave,
            self.M[3][3] * 2 * self.zeta_roll  * self.w_roll,
            self.M[4][4] * 2 * self.zeta_pitch * self.w_pitch,
            self.M[5][5] / self.T_yaw
            ])
        # Linear surge and sway damping
        D[0][0] = D[0][0] * np.exp(-3*U_r) # vanish at high speed where quadratic
        D[1][1] = D[1][1] * np.exp(-3*U_r) # drag and lift forces dominates

        ### Hydrodynamic Forces ###
        tau_liftdrag = self.forceLiftDrag(self.diam,self.S,self.CD_0,alpha,U_r)
        tau_crossflow = self.crossFlowDrag(self.L,self.diam,self.diam,nu_r)

        ### Restoring Forces ###
        g = self.gvect(self.W, self.B, self.eta[4], self.eta[3], self.r_bg, self.r_bb)

        ### Control Forces ###
        tau_control = self.control_forces(self.u_actual, nu_r, U)
        
        
        ############### COMPUTE DYNAMICS #################
        tau_sum = tau_control + tau_liftdrag + tau_crossflow - np.matmul(C+D,nu_r) - g
        nu_dot = Dnu_c + np.matmul(self.Minv, tau_sum) # Acceleration from forces plus ocean current acceleration

        # Move the actuators towards commanded value & saturate #TODO: saturating isn't being down here
        euler_u, u_actual_dot = self.actuator_dynamics(sampleTime, u_control, self.u_actual) 


        return nu_dot, u_actual_dot
    
    def control_forces(self, u_actual, nu_r, U):
        """
        Vehicle-specific calculations for dynamics of the actuators (fins and thruster).

        Note: For Torpedo Vehicles, positive fin deflection will pitch the vehicle up and yaw to the starboard side.

        :param array-like u_actual: Current control surface position.
        :param array-like nu_r: Reference velocity of the vehicle in the body frame.

        :returns: two arrays: forces and moments
        """

        ################### FIN CALCULATIONS #########################
        delta_r = u_actual[0]       # actual tail rudder (rad)
        delta_re = u_actual[1]      # actual right elevator (rad)
        delta_le = u_actual[2]      # actual left elevator (rad)
        n = u_actual[-1]            # Propellor speed (rpm)

        # Horizontal- and vertical-plane relative speed   
        U_rh = np.sqrt(nu_r[0]**2 + nu_r[1]**2)
        U_re = np.sqrt(nu_r[0]**2 + (nu_r[1] * np.sin(self.D2R * 30))**2 + (nu_r[2] * np.sin(self.D2R * 60))**2)

        #Positive rudder deflection turn the vehicle right, positive elevator deflection pitches vehicle up

        #lift forces on the elevator fins on right and left both positive set direction below
        fl_re = 0.5 * self.rho * U_re**2 * self.S_fin * self.CL_delta_s * delta_re
        fl_le = 0.5 * self.rho * U_re**2 * self.S_fin * self.CL_delta_s * delta_le

        # Rudder and elevator drag [TODO: these don't look right to me...]
        X_r = -0.5 * self.rho * U_rh**2 * self.S_fin * self.CL_delta_r * delta_r**2
        X_re = -0.5 * self.rho * U_re**2 * self.S_fin * self.CL_delta_s * delta_re**2
        X_le = -0.5 * self.rho * U_re**2 * self.S_fin * self.CL_delta_s * delta_le**2

        # Rudder and elevator sway force (Positive deflection -> negative Y force -> positive Z moment (yaw right))
        Y_r = -0.5 * self.rho * U_rh**2 * self.S_fin * self.CL_delta_r * delta_r
        Y_re = -fl_re * np.sin(30 * self.D2R)
        Y_le = fl_le * np.sin(30 * self.D2R)  

        # elevator heave force  (positive z force)
        Z_re = fl_re * np.sin(60 * self.D2R)     
        Z_le = fl_le * np.sin(60 * self.D2R)

        ################# Propeller Calulations ################
        
        # Propeller coeffs. KT and KQ are computed as a function of advance no.
        # Ja = Va/(n*D_prop) where Va = (1-w)*U = 0.944 * U; Allen et al. (2000)
        n_rps = n / 60  # propeller revolution (rps) 
        Va = self.Va * U  # advance speed (m/s)

        # Ja_max = 0.944 * 2.5 / (0.14 * 1525/60) = 0.6632
        
        
        # Single-screw propeller with 3 blades and blade-area ratio = 0.718.
        # Coffes. are computed using the Matlab MSS toolbox:     
        # >> [KT_0, KQ_0] = wageningen(0,1,0.718,3)
        # >> [KT_max, KQ_max] = wageningen(0.6632,1,0.718,3) 
        
        
        # Propeller thrust and propeller-induced roll moment
        # Linear approximations for positive Ja values
        # KT ~= KT_0 + (KT_max-KT_0)/Ja_max * Ja   
        # KQ ~= KQ_0 + (KQ_max-KQ_0)/Ja_max * Ja  
      
        if n_rps > 0:   # forward thrust
            X_prop = self.rho * pow(self.D_prop,4) * ( 
                self.KT_0 * abs(n_rps) * n_rps + (self.KT_max-self.KT_0)/self.Ja_max * 
                (Va/self.D_prop) * abs(n_rps) )        
            K_prop = self.rho * pow(self.D_prop,5) * (
                self.KQ_0 * abs(n_rps) * n_rps + (self.KQ_max-self.KQ_0)/self.Ja_max * 
                (Va/self.D_prop) * abs(n_rps) )           
        else:    # reverse thrust (braking)
            X_prop = self.rho * pow(self.D_prop,4) * self.KT_0 * abs(n_rps) * n_rps 
            K_prop = self.rho * pow(self.D_prop,5) * self.KQ_0 * abs(n_rps) * n_rps 

        fx = X_r + X_re + X_le + (1-self.t_prop) * X_prop
        fy = Y_r + Y_re + Y_le
        fz = Z_le + Z_re
        Mx = self.fin_center * (fl_re - fl_le - Y_r)  # Rolling moment from the fins
        My = -self.x_fin * fz # -1 comes from the cross product of x with z                           
        Mz =  self.x_fin * (Y_r + Y_re + Y_le) + K_prop/10

        tau = np.array([fx, fy, fz, Mx, My, Mz], float)
        return tau

    def actuator_dynamics(self, sampleTime, u_control, u_actual):
        u_actual_dot = []

        #Fin Speed
        for i in range(len(u_control)-1):
            u_actual_dot.append((u_control[i] - u_actual[i]) / self.T_delta) 

        #Thruster acceleration
        u_actual_dot.append((u_control[-1] - u_actual[-1]) / self.T_n) 

        #Control surface integration
        for i in range(len(u_control)):
            u_actual[i] += sampleTime * u_actual_dot[i]

        return u_actual, u_actual_dot

    def saturate_actuator(self,u_actual):

        #Saturate fins
        for i in range(len(u_actual)-1):
            # Amplitude saturation of the control signals
            if abs(u_actual[i]) >= self.deltaMax_r:
                u_actual[i] = np.sign(u_actual[i]) * self.deltaMax_r

        # Saturate thruster value  
        if abs(u_actual[-1]) >= self.nMax:
            u_actual[-1] = np.sign(u_actual[-1]) * self.nMax 

        return u_actual
    
    # TODO: Write this function
    def step(self, command, timestep, method='euler'):
        # TODO: Write a Runge-Kutta 4 thing to integrate the state and actuator. 
        # Calcuate the new velocities nu and position eta and the new control positions. 

        # self.nu = 1
        # self.eta = 1
        # self.u_actual = 1
        nu = self.nu.copy() #array of size 6
        eta = self.eta.copy() #array of size 6
        u_actual = self.u_actual.copy() #array of size 4
        prior = np.concatenate((eta,nu,u_actual))

        if method == 'euler':
            next_nu_dot, next_u_actual_dot = self.dynamics(command,timestep)
            statedot = np.concatenate(nu,next_nu_dot,next_u_actual_dot)
            newState = self.stateEulerStep(prior,statedot,timestep)
            self.eta = newState[:6].copy()
            self.nu = newState[6:12].copy()
            self.u_actual = self.saturate_actuator(newState[12:])
        elif method == 'rk4':
            raise NotImplementedError()
        elif method == 'rk3':
            raise NotImplementedError()
        else:
            raise ValueError("method: {} not found, please use euler, rk3 or rk4".format(method))
        
    #### Helper functions ####

    #does this one need to be in the class
    def Hmtrx(self, r):
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

    #might not need to be in the class
    def attitudeEuler(self, eta, nu, sampleTime):
        """
        eta = attitudeEuler(eta,nu,sampleTime) computes the generalized 
        position/Euler angles eta[k+1]
        """
    
        p_dot   = np.matmul(Rzyx(eta[3], eta[4], eta[5]), nu[0:3] )
        v_dot   = np.matmul(Tzyx(eta[3], eta[4]), nu[3:6] )

        # Forward Euler integration
        output = np.zeros_like(eta)
        output = eta[0:3] + sampleTime * p_dot
        output = eta[3:6] + sampleTime * v_dot

        return output
    
    def stateEulerStep(self, state, state_dot, timeStep):
        new_state = np.zeros_like(state)
        state_dot_transformed = state_dot.copy()
        state_dot_transformed[:6] = velocityTransform(state[:6],state_dot[0:6])
        new_state = state + timeStep * state_dot_transformed
        return new_state

    def m2c(self, M, nu):
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
            dt_dnu1 = np.matmul(M11,nu1) + np.matmul(M12,nu2)
            dt_dnu2 = np.matmul(M21,nu1) + np.matmul(M22,nu2)

            #C  = [  zeros(3,3)      -Smtrx(dt_dnu1)
            #      -Smtrx(dt_dnu1)  -Smtrx(dt_dnu2) ]
            C = np.zeros( (6,6) )    
            C[0:3,3:6] = -self.Smtrx(dt_dnu1)
            C[3:6,0:3] = -self.Smtrx(dt_dnu1)
            C[3:6,3:6] = -self.Smtrx(dt_dnu2)
                
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

    def forceLiftDrag(self, b,S,CD_0,alpha,U_r):
        """
        tau_liftdrag = forceLiftDrag(b,S,CD_0,alpha,Ur) computes the hydrodynamic
        lift and drag forces of a submerged "wing profile" for varying angle of
        attack (Beard and McLain 2012). Application:
        
        M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_liftdrag
        
        Inputs:
            b:     wing span (m)
            S:     wing area (m^2)
            CD_0:  parasitic drag (alpha = 0), typically 0.1-0.2 for a streamlined body
            alpha: angle of attack, scalar or vector (rad)
            U_r:   relative speed (m/s)

        Returns:
            tau_liftdrag:  6x1 generalized force vector
        """
        raise NotImplementedError

        #helper function
        def coeffLiftDrag(b,S,CD_0,alpha,sigma):
            
            """
            [CL,CD] = coeffLiftDrag(b,S,CD_0,alpha,sigma) computes the hydrodynamic 
            lift CL(alpha) and drag CD(alpha) coefficients as a function of alpha
            (angle of attack) of a submerged "wing profile" (Beard and McLain 2012)

            CD(alpha) = CD_p + (CL_0 + CL_alpha * alpha)^2 / (pi * e * AR)
            CL(alpha) = CL_0 + CL_alpha * alpha
    
            where CD_p is the parasitic drag (profile drag of wing, friction and
            pressure drag of control surfaces, hull, etc.), CL_0 is the zero angle 
            of attack lift coefficient, AR = b^2/S is the aspect ratio and e is the  
            Oswald efficiency number. For lift it is assumed that
    
            CL_0 = 0
            CL_alpha = pi * AR / ( 1 + sqrt(1 + (AR/2)^2) );
    
            implying that for alpha = 0, CD(0) = CD_0 = CD_p and CL(0) = 0. For
            high angles of attack the linear lift model can be blended with a
            nonlinear model to describe stall
    
            CL(alpha) = (1-sigma) * CL_alpha * alpha + ...
                sigma * 2 * sign(alpha) * sin(alpha)^2 * cos(alpha) 

            where 0 <= sigma <= 1 is a blending parameter. 
            
            Inputs:
                b:       wing span (m)
                S:       wing area (m^2)
                CD_0:    parasitic drag (alpha = 0), typically 0.1-0.2 for a 
                        streamlined body
                alpha:   angle of attack, scalar or vector (rad)
                sigma:   blending parameter between 0 and 1, use sigma = 0 f
                        or linear lift 
                display: use 1 to plot CD and CL (optionally)
            
            Returns:
                CL: lift coefficient as a function of alpha   
                CD: drag coefficient as a function of alpha   

            Example:
                Cylinder-shaped AUV with length L = 1.8, diameter D = 0.2 and 
                CD_0 = 0.3
                
                alpha = 0.1 * pi/180
                [CL,CD] = coeffLiftDrag(0.2, 1.8*0.2, 0.3, alpha, 0.2)
            """
            
            e = 0.7             # Oswald efficiency number
            AR = b**2 / S       # wing aspect ratio

            # linear lift
            CL_alpha = np.pi * AR / ( 1 + np.sqrt(1 + (AR/2)**2) )
            CL = CL_alpha * alpha

            # parasitic and induced drag
            CD = CD_0 + CL**2 / (np.pi * e * AR)
            
            # nonlinear lift (blending function)
            CL = (1-sigma) * CL + sigma * 2 * np.sign(alpha) \
                * np.sin(alpha)**2 * np.cos(alpha)

            return CL, CD

        
        [CL, CD] = coeffLiftDrag(b,S,CD_0,alpha,0) 
        
        F_drag = 1/2 * self.rho * U_r**2 * S * CD    # drag force
        F_lift = 1/2 * self.rho * U_r**2 * S * CL    # lift force

        # transform from FLOW axes to BODY axes using angle of attack
        tau_liftdrag = np.array([
            np.cos(alpha) * (-F_drag) - np.sin(alpha) * (-F_lift),
            0,
            np.sin(alpha) * (-F_drag) + np.cos(alpha) * (-F_lift),
            0,
            0,
            0 ])

        return tau_liftdrag

def velocityTransform(eta, nu):
    #transform the velocity
    #modeled after attitudeEuler
    #but doesnt do euler step, for use w/ rk3 or rk4
    p_dot = np.matmul(Rzyx(eta[3],eta[4],eta[5]),nu[:3])
    temp = nu[3:6].copy()
    temp.resize(3,1)
    v_dot = np.matmul(Tzyx(eta[3],eta[4]),nu[3:6])
    return np.append(p_dot,v_dot)

#Todo: move to a helper file
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

def crossFlowDrag(L,B,T,nu_r):
    """
    tau_crossflow = crossFlowDrag(L,B,T,nu_r) computes the cross-flow drag 
    integrals for a marine craft using strip theory. 

    M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_crossflow
    """

    rho = 1026               # density of water
    n = 20                   # number of strips

    dx = L/20             
    Cd_2D = Hoerner(B,T)    # 2D drag coefficient based on Hoerner's curve

    Yh = 0
    Nh = 0
    xL = -L/2
    
    for i in range(0,n+1):
        v_r = nu_r[1]             # relative sway velocity
        r = nu_r[5]               # yaw rate
        Ucf = abs(v_r + xL * r) * (v_r + xL * r)
        Yh = Yh - 0.5 * rho * T * Cd_2D * Ucf * dx         # sway force
        Nh = Nh - 0.5 * rho * T * Cd_2D * xL * Ucf * dx    # yaw moment
        xL += dx
        
    tau_crossflow = np.array([0, Yh, 0, 0, 0, Nh],float)

    return tau_crossflow

def gvect(W,B,theta,phi,r_bg,r_bb):
        """
        g = gvect(W,B,theta,phi,r_bg,r_bb) computes the 6x1 vector of restoring 
        forces about an arbitrarily point CO for a submerged body. 
        
        Inputs:
            W, B: weight and buoyancy (kg)
            phi,theta: roll and pitch angles (rad)
            r_bg = [x_g y_g z_g]: location of the CG with respect to the CO (m)
            r_bb = [x_b y_b z_b]: location of the CB with respect to th CO (m)
            
        Returns:
            g: 6x1 vector of restoring forces about CO
        """

        sth  = np.sin(theta)
        cth  = np.cos(theta)
        sphi = np.sin(phi)
        cphi = np.cos(phi)

        g = np.array([
            (W-B) * sth,
            -(W-B) * cth * sphi,
            -(W-B) * cth * cphi,
            -(r_bg[1]*W-r_bb[1]*B) * cth * cphi + (r_bg[2]*W-r_bb[2]*B) * cth * sphi,
            (r_bg[2]*W-r_bb[2]*B) * sth         + (r_bg[0]*W-r_bb[0]*B) * cth * cphi,
            -(r_bg[0]*W-r_bb[0]*B) * cth * sphi - (r_bg[1]*W-r_bb[1]*B) * sth      
            ])
        
        return g

def ssa(angle):
    """
    angle = ssa(angle) returns the smallest-signed angle in [ -pi, pi )
    """
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
        
    return angle 

def sat(x, x_min, x_max):
    """
    x = sat(x,x_min,x_max) saturates a signal x such that x_min <= x <= x_max
    """
    if x > x_max:
        x = x_max 
    elif x < x_min:
        x = x_min
        
    return x 

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