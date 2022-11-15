#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from numpy import cos, sin


# In[38]:


class PowerFlow:
    
    def __init__(self, Ybarra, Swing, PV, PQ, max_error = 1e-3, ini_method = 'flat_start', save = False, path = ''):
        
        self.Ybarra     = Ybarra
        self.Swing      = Swing
        self.PV         = PV
        self.PQ         = PQ
        self.maxerror   = max_error
        self.ini_method = str.lower(ini_method)
        self.start      = {'flat_start': self.FlatStart,
                          }

        try:
            self.ini_theta, self.ini_V = self.start[self.ini_method]()
        except KeyError:
            print(ini_method + ' was not recognized as a start mathod.')

        self.set_V(array_idx = np.concatenate([[Swing[0]], PV[:,0], PQ[:,0]]), array_values = np.concatenate([[Swing[1]], PV[:,2], self.ini_V]))
        self.set_theta(array_idx = np.concatenate([[Swing[0]], PV[:,0], PQ[:,0]]), array_values = np.concatenate([[Swing[2]], self.ini_theta]))
        self.loop()
        
        if save:
            np.save(path + 'convergence_report.npy', self.history, allow_pickle = True)
        
    def loop(self):

        run_shot = {}
        self.count = 0
        # Mismatch calculation
        P_esp = np.concatenate([self.PV[:,1],self.PQ[:,1]])
        Q_esp = self.PQ[:,2]

       
        P_calc, Q_calc = self.get_PQ(array_V = self.get_V(), array_theta = self.get_theta()) # Calculate PQ of certain buses considering their V and theta
        
        P_mismatch = P_esp - P_calc[1:,1] # Take all P_calc but in the Swing Bus, which is in the 0 idx
        Q_mismatch = Q_esp - Q_calc[-self.PQ.shape[0]:,1] # Take all Q_calc in the PQ Busses

        
        self.ScreenShot(P_calc, Q_calc, P_mismatch, Q_mismatch)
        

        while(self.count < 10 and np.max(np.abs(np.concatenate([P_mismatch, Q_mismatch]))) > self.maxerror):

            J = self.Jacob(array_V = self.get_V(), array_theta = self.get_theta(), P_calc = P_calc, Q_calc = Q_calc)

            delta_theta_V = np.linalg.inv(J)@np.concatenate([P_mismatch, Q_mismatch])

            self.set_theta(array_idx =  np.concatenate([self.PV[:,0], self.PQ[:,0]]),
                        array_values = self.get_theta()[1:, 1] + delta_theta_V[:self.get_theta()[1:, 1].shape[0]])
            
            self.set_V(array_idx =  self.PQ[:,0],
                    array_values = self.get_V()[-self.PQ.shape[0]:, 1] + delta_theta_V[-self.PQ.shape[0]:] )
            
            self.count += 1
            P_calc, Q_calc = self.get_PQ(array_V = self.get_V(), array_theta = self.get_theta()) # Calculate PQ of certain buses considering their V and theta
            
            P_mismatch = P_esp - P_calc[1:,1] # Take all P_calc but in the Swing Bus, which is in the 0 idx
            Q_mismatch = Q_esp - Q_calc[-self.PQ.shape[0]:,1] # Take all Q_calc in the PQ Busses

            self.ScreenShot(P_calc, Q_calc, P_mismatch, Q_mismatch)
        

        

    def Jacob(self, array_V, array_theta, P_calc, Q_calc):

        # H block and N block
        H = []
        N = []
        
        for k, Pk in P_calc[1:]:

            Vk = array_V[array_V[:,0] == k, 1]
            theta_k = array_theta[array_theta[:,0] == k, 1]

            for m, theta_m in array_theta[1:]:

                Vm = array_V[array_V[:,0] == m,1]
                Gkm = np.real(self.Ybarra[int(k) - 1, int(m) - 1])
                Bkm = np.imag(self.Ybarra[int(k) - 1, int(m) - 1])

                if k != m:
                    H.append( Vk*Vm*(Gkm*sin(theta_k - theta_m) - Bkm*cos(theta_k - theta_m)) )
                else:
                    H.append( -Bkm*Vk**2 - Q_calc[Q_calc[:,0] == k, 1] )

            for m, Vm in array_V[-self.PQ.shape[0]:]:

                theta_m = array_theta[array_theta[:,0] == m,1]
                Gkm = np.real(self.Ybarra[int(k) - 1, int(m) - 1])
                Bkm = np.imag(self.Ybarra[int(k) - 1, int(m) - 1])

                if k != m:
                    N.append( Vk*(Gkm*cos(theta_k - theta_m) + Bkm*sin(theta_k - theta_m)) )
                else:
                    N.append( Gkm*Vk + Pk/Vk )

        H = np.array(H).reshape((self.PQ.shape[0] + self.PV.shape[0], self.PQ.shape[0] + self.PV.shape[0]))
        N = np.array(N).reshape((self.PQ.shape[0] + self.PV.shape[0], self.PQ.shape[0]))

        # M and L blocks
        M = []
        L = []

        for k, Qk in Q_calc[-self.PQ.shape[0]:]:

            Vk = array_V[array_V[:,0] == k, 1]
            theta_k = array_theta[array_theta[:,0] == k, 1]

            # M
            for m, theta_m in array_theta[1:]:

                Vm = array_V[array_V[:,0] == m, 1]
                Gkm = np.real(self.Ybarra[int(k) - 1, int(m) - 1])
                Bkm = np.imag(self.Ybarra[int(k) - 1, int(m) - 1])

                if k != m:
                    M.append( -Vk*Vm*(Gkm*cos(theta_k - theta_m) + Bkm*sin(theta_k - theta_m)) )
                else:
                    M.append( -Gkm*Vk**2 + P_calc[P_calc[:,0] == k, 1] )
                    
            for m, Vm in array_V[-self.PQ.shape[0]:]:

                theta_m = array_theta[array_theta[:,0] == m, 1]
                Gkm = np.real(self.Ybarra[int(k) - 1, int(m) - 1])
                Bkm = np.imag(self.Ybarra[int(k) - 1, int(m) - 1])

                if k != m:
                    L.append( Vk*(Gkm*sin(theta_k - theta_m) - Bkm*cos(theta_k - theta_m)) )
                else:
                    L.append( -Bkm*Vk + Qk/Vk )

        M = np.array(M).reshape((self.PQ.shape[0], self.PQ.shape[0] + self.PV.shape[0]))
        L = np.array(L).reshape((self.PQ.shape[0], self.PQ.shape[0]))

        J = np.block([[H,N],
                  [M,L]])
        return J    
        
        
    def get_PQ(self, array_V, array_theta):

        if array_V.shape == array_theta.shape:

            P_calc = []
            Q_calc = []
            
            for k, theta_k in array_theta:

                Vk = array_V[array_V[:,0] == k, 1][0]
                P_aux = 0
                Q_aux = 0

                for m, Vm in array_V:
                                       
                    theta_m = array_theta[array_theta[:,0] == m, 1][0]
                    Gkm = np.real(self.Ybarra[int(k) - 1, int(m) - 1])
                    Bkm = np.imag(self.Ybarra[int(k) - 1, int(m) - 1])

                    P_aux += Vk*Vm*(Gkm*cos(theta_k - theta_m) + Bkm*sin(theta_k - theta_m))
                    Q_aux += Vk*Vm*(Gkm*sin(theta_k - theta_m) - Bkm*cos(theta_k - theta_m))
                
                P_calc.append([k, P_aux])
                Q_calc.append([k, Q_aux])
            
            return np.array(P_calc), np.array(Q_calc)

        else:
            raise ValueError('Voltage and phase vectors are not the same size.')

    def set_V(self, array_idx, array_values):

        try:
            list_idx = [np.where(self.array_V[:,0] == idx)[0][0] for idx in array_idx]
            self.array_V[list_idx,1] = array_values

        except AttributeError:
            self.array_V = np.transpose(np.concatenate([[array_idx], [array_values]]))

    def get_V(self, array_idx = []):

        if array_idx != []:
            return self.array_V[[np.where(self.array_V[:,0] == idx)[0][0] for idx in array_idx]]
        else: return self.array_V

    def set_theta(self, array_idx, array_values):

        try:
            list_idx = [np.where(self.array_theta[:,0] == idx)[0][0] for idx in array_idx]
            self.array_theta[list_idx, 1] = array_values

        except AttributeError:
            self.array_theta = np.transpose(np.concatenate([[array_idx], [array_values]]))

    def get_theta(self, array_idx = []):
            if array_idx != []:
                return self.array_theta[[np.where(self.array_theta[:,0] == idx)[0][0] for idx in array_idx]]
            else: return self.array_theta
    
    def ActivePower(self, Vk, Vm, theta_k, theta_m, gkm, bkm):

        return Vk*Vm*(gkm*cos(theta_k - theta_m) + bkm*sin(theta_k - theta_m))
        
    def ReactivePower(self, Vk, Vm, theta_k, theta_m, gkm, bkm):
        return Vk*Vm*(gkm*sin(theta_k - theta_m) - bkm*cos(theta_k - theta_m))
    
    def FlatStart(self):
        
        ini_theta = np.zeros((self.PV.shape[0] + self.PQ.shape[0]))
        ini_V = np.ones((self.PQ.shape[0]))
        
        return ini_theta, ini_V
    
    def ScreenShot(self, Pc, Qc, dP, dQ):

        try:
            self.history['V'] = np.vstack((self.history['V'], self.get_V()[:,1]))
            self.history['theta'] = np.vstack((self.history['theta'], self.get_theta()[:,1]))
            self.history['Pcalc'] = np.vstack((self.history['Pcalc'], Pc[:,1]))
            self.history['Qcalc'] = np.vstack((self.history['Qcalc'], Qc[:,1]))
            self.history['dP'] = np.vstack((self.history['dP'], dP))
            self.history['dQ'] = np.vstack((self.history['dQ'], dQ))

        except AttributeError:
            self.history = {'V':            self.get_V()[:,1],
                            'theta':        self.get_theta()[:,1],
                            'Pcalc':        Pc[:,1],
                            'Qcalc':        Qc[:,1],
                            'dP':           dP,
                            'dQ':           dQ,
                            'error':        self.maxerror,
                            'bus':          self.get_V()[:,0],
                    }


# In[34]:


# Entrar com a Ybarra, vetor de incialização de tensões e fases

y12 = -1j/0.015
y23 = 2/(0.0009 + 0.014j)
y34 = 1/(0.015j)
y2 = 0.6j

Ybarra = np.array([[ y12,  -y12,      0,      0 ],
                   [-y12, y12+y23+2*y2,  -y23,     0 ],
                   [  0,   -y23,   y23+y34+2*y2, -y34],
                   [  0,     0,     -y34,    y34]])
      
# Definicao dos tipos de barra com respeito às linhas de Ybarra
# Ex: Swing [a x y] -> barra a, V = 1, theta = 0
#     PV [a x y] -> barra a, P = x, V = y
#     PQ [a x y] -> barra a, P = x, Q = y

Swing = np.array([4, 1, 0])
PV    = np.array([[1, 4, 1],])
PQ    = np.array([[2,  0, 0],
                  [3, -2, -1]]) 

      


# In[39]:


pf = PowerFlow(Ybarra, Swing, PV, PQ, 5e-5, 'flat_start', True)

