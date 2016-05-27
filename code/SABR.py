# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 10:50:13 2016

By using SABR, fitting implied vol and surface, construction local surface

@author: Minhyun Yoo
"""  

#param = [alp, bet, rho, nu];

# trick for using some variables in function like global variables
def SetVar( input_s, input_Tau, input_r, input_q ):
    global s, Tau, r, q;
    s = input_s; Tau = input_Tau;
    r = input_r; q = input_q;

#def GetVar():
#    global g_s, g_Tau, g_r, g_q;
#    return g_s, g_Tau, g_r, g_q;

def SABR_func( K, alp, bet, rho, nu ): 
    import numpy as np;
#    from math import isnan, exp, log, sqrt;
#    s, Tau, r, q
    
    f = s*np.exp((r - q)*Tau);
    
    z = nu/alp*(f * K)**(0.5*(1 - bet)) * np.log(f / K);
    xz = np.log((np.sqrt(1 - 2*rho*z + z*z) + z - rho) / (1 - rho));    
    
    zdivxz = z / xz;
    
    # exception cases
    zdivxz[np.isnan(zdivxz)] = 1.0;
    
    result = ( alp*(f*K)**(0.5* (bet-1) )*
        (1 + ( ((1 - bet)*np.log(f/K))**2/24 + ((1 - bet)*np.log(f/K))**4/1920 ))**(-1.0)
        * zdivxz
        * ( 1 + ( ((1 - bet)*alp )**2 / (24*(f*K)**(1 - bet)) 
        + 0.25*alp*bet*rho*nu / ((f*K)**(0.5*(1-bet))) 
        + ((2-3*rho**2)*nu**2)/24)*Tau ) 
        );
        
    return result;
    
    
    