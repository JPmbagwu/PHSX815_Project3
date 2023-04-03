#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 19:28:29 2023

@author: johnpaulmbagwu
"""

# Python code for 2D random walk. 
import sys
import numpy
import matplotlib.pyplot as plt
import pylab
from scipy.stats import chi2
#from random import random
import math
#from scipy.special import factorial
sys.path.append(".")
import Random as rng




if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv:
        print ("Usage: %s [-seed number] [-Nwalk number] [-Nstep number] [-Lstep] [-CL number]" % sys.argv[0])
        print
        sys.exit(1)

    # default seed
    seed = 5565
    
    # default Nwalk
    Nwalk = 2000
    
    # default Nstep
    Nstep = 300
    
    # default Lstep
    Lstep = 1.

    # default Confidence level
    CL = 0.78

    # read the user-provided seed from the command line (if there)
    if '-seed' in sys.argv:
        p = sys.argv.index('-seed')
        seed = sys.argv[p+1]    
    if '-Nwalk' in sys.argv:
        p = sys.argv.index('-Nwalk')
        Nw = int(sys.argv[p+1])
        if Nw > 0:
            Nwalk = Nw
    if '-Nstep' in sys.argv:
        p = sys.argv.index('-Nstep')
        Ns = int(sys.argv[p+1])
        if Ns > 0:
            Nstep = Ns
    if '-Lstep' in sys.argv:
        p = sys.argv.index('-Lstep')
        Ls = float(sys.argv[p+1])
        if Ls > 0:
            Lstep = Ls
    if '-CL' in sys.argv:
        p = sys.argv.index('-CL')
        cl = float(sys.argv[p+1])
        if 0. < cl < 1.:
            CL = cl
    # class instance of our Random class using seed
    random = rng.Random(seed)
    
    #############################################################
    ################ RANDOM WALK #######################
    #############################################################
    # Define positions and displacement in arrays
    xfinal = []
    yfinal = []
    finald = []
    finald2 = []
######## PERFORM THE RANDOM WALK ################################# 
    for j in range(1,Nwalk+1):
        x = numpy.zeros(Nstep) 
        y = numpy.zeros(Nstep)
        for i in range(1, Nstep):
            z1, z2 = random.BoxMuller(Lstep)
            x[i] = x[i - 1] + z1
            y[i] = y[i - 1] + z2
        finald.append(numpy.sqrt(x[Nstep-1]*x[Nstep-1]+y[Nstep-1]*y[Nstep-1]))
        finald2.append(x[Nstep-1]*x[Nstep-1]+y[Nstep-1]*y[Nstep-1])
        xfinal.append(x[Nstep-1])
        yfinal.append(y[Nstep-1])
#########  PARAMETER ESTIMATION ############################
    addt = 0.
    for j in range(0,Nwalk):
        addt = addt + finald2[j]
        par = 1./(2.*Nwalk) * addt
############# ERRORS ######################################
    alph = 1. - CL
    avr2 = sum(finald2)/Nwalk
    CLmin = Nwalk*avr2/chi2.ppf(1.-alph/2. , 2*Nwalk)
    CLmax = Nwalk*avr2/chi2.ppf(alph/2. , 2*Nwalk)
################################################################
##############PRINT RESULTS ###################################
    print('############################################################')
    print('                                                           #')
    print('Estimated parameter: ', round(par,15) ,'                   #')
    print('C.L. ', CL*100, '% in [',round(CLmin,15),',',round(CLmax,15),']  #' )
    print('                                                           #')
    print('############################################################')
############### RANDOM WALK PLOTS#########################
################################################################
    plt.scatter(xfinal, yfinal, s = 0.5 , alpha = 0.5 , color='steelblue')
    plt.grid(axis='x', alpha=0.75)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Final positions after ' + str(Nwalk) + ' walks and ' + str(Nstep) + ' steps')
    plt.show()
################################################################
    f1 = numpy.linspace(0.0,max(finald),1000)
    f2 = f1/par * numpy.exp(-f1*f1/(2.*par))
    plt.plot(f1,f2, color='black', linestyle='dashed', label = 'Rayleigh distribution, $ \sigma^2$= ' + str(round(par,3)) + ' C.L. '
                 + str(CL*100) + '% in [' + str(round(CLmin,3)) +',' + str(round(CLmax,3)) + ']' )
    n, bins, patches = plt.hist(finald, 80, color ='steelblue', density = True, alpha=0.6, fill = True, hatch='/',histtype='step', linewidth=1, label = 'data')
    plt.legend(loc='upper right')
    plt.title('Data distribution in comparison to the Rayleigh distribution ')
    plt.xlabel('r')
    plt.ylabel('Probability')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
