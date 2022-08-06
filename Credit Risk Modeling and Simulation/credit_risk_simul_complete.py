# Import libraries
import numpy as np
import pandas as pd
import scipy
import math
from scipy import special
from pathlib import Path
import scipy.stats as scs
from scipy.stats import norm
#To know how long your code is taking to run
from datetime import datetime
from pytz import timezone
import time
#To plot your results
import matplotlib.pyplot as plt

Nout = 100000  # number of out-of-sample scenarios
Nin = 5000     # number of in-sample scenarios
Ns = 5         # number of idiosyncratic scenarios for each systemic

C = 8          # number of credit states

# Read and parse instrument data
instr_data = np.array(pd.read_csv('instrum_data.csv', header=None))
instr_id = instr_data[:, 0]     # ID
driver = instr_data[:, 1]       # credit driver
beta = instr_data[:, 2]         # beta (sensitivity to credit driver)
recov_rate = instr_data[:, 3]   # expected recovery rate
value = instr_data[:, 4]        # value
prob = instr_data[:, 5:(5 + C)] # credit-state migration probabilities (default to AAA)
exposure = instr_data[:, 5 + C:5 + 2 * C]  # Loss in credit-state migration (default to AAA)
retn = instr_data[:, 5 + 2 * C] # market returns

K = instr_data.shape[0]         # number of CounterParties

# Read matrix of correlations for credit drivers
rho = np.array(pd.read_csv('credit_driver_corr.csv', sep='\t', header=None))
# Cholesky decomp of rho (for generating correlated Normal random numbers)
sqrt_rho = np.linalg.cholesky(rho)

print('======= Credit Risk Model with Credit-State Migrations =======')
print('============== Monte Carlo Scenario Generation ===============')
print(' ')
print(' ')
print(' Number of out-of-sample Monte Carlo scenarios = ' + str(Nout))
print(' Number of in-sample Monte Carlo scenarios = ' + str(Nin))
print(' Number of counterparties = ' + str(K))
print(' ')


# Find credit-state for each counterparty
# 8 = AAA, 7 = AA, 6 = A, 5 = BBB, 4 = BB, 3 = B, 2 = CCC, 1 = default
CS = np.argmax(prob, axis=1) + 1

# Account for default recoveries
exposure[:, 0] = (1 - recov_rate) * exposure[:, 0]

# Compute credit-state boundaries
CS_Bdry = scipy.special.ndtri((np.cumsum(prob[:, 0:C - 1], 1)))

# -------- Insert your code here -------- #
NumRF = rho.shape[0] # number of risk factors(credit drivers)
filename_save_out = 'scen_out'
if Path(filename_save_out+'.npz').is_file():
    Losses_out = scipy.sparse.load_npz(filename_save_out + '.npz')
else:
    # Generating Scenarios
    # create y to store systematic risk components
    y = np.zeros((Nout, NumRF)) # for 100000 scenarios and 50 credit drivers
    # create w to store creditworthiness index
    #w = np.zeros((Nout, K)) # for 100000 scenarios and 100 counter parties
    # create Losses_out to store losses
    Losses_out = np.zeros((Nout, K))# for 100000 scenarios and 100 counter parties
    # define idiosyncratic component for 100 counter parties from N(0,1)
    z = np.random.normal(0, 1, (K,1))

    for s in range(1, Nout + 1):
        # -------- Insert your code here -------- #
        random_norm = np.random.normal(0,1, (NumRF, 1)) #from N(0,1) 
        #generate correlated Normal random numbers using Cholesky decomp of rho
        y[s-1,:] = np.dot(sqrt_rho,random_norm).T 
        for cp in range(1, K + 1):
            RF = int(driver[cp-1]) #get credit driver index for that counter party
            #Calculate creditworthiness index
            w = beta[cp-1] * y[s-1, RF-1] + np.sqrt(1 - beta[cp-1]**2) * z[cp-1]
            # append w to CS boundaries, and sort the values
            w_CS_Bdry= np.sort(np.append(w, CS_Bdry[cp-1,:]))
            #find credit state index
            CS_index=np.argwhere(w_CS_Bdry == w)
            # Calculated out-of-sample losses (100000 x 100)
            Losses_out[s-1,cp-1] = exposure[cp-1, CS_index]
    # Losses_out as Compressed Sparse Row matrix     
    Losses_out = scipy.sparse.csr_matrix(Losses_out)
    scipy.sparse.save_npz(filename_save_out + '.npz', Losses_out)

# Normal approximation computed from out-of-sample scenarios
mu_l = np.mean(Losses_out, axis=0).reshape((K))
var_l = np.cov(Losses_out.toarray(), rowvar=False) # Losses_out as a sparse matrix

# Compute portfolio weights
portf_v = sum(value)  # portfolio value
w0 = []
w0.append(value / portf_v)   # asset weights (portfolio 1)
w0.append(np.ones((K)) / K)  # asset weights (portfolio 2)
x0 = []
x0.append((portf_v / value) * w0[0])  # asset units (portfolio 1)
x0.append((portf_v / value) * w0[1])  # asset units (portfolio 2)

# Quantile levels (99%, 99.9%)
alphas = np.array([0.99, 0.999])

VaRout = np.zeros((2, alphas.size))
VaRinN = np.zeros((2, alphas.size))
CVaRout = np.zeros((2, alphas.size))
CVaRinN = np.zeros((2, alphas.size))
for portN in range(2):
    # Compute VaR and CVaR
    # Sort losses in increasing order
    Loss_port = np.sort(np.dot(Losses_out.toarray(),x0[portN]))
    for q in range(alphas.size):
        alf = alphas[q]
        # -------- Insert your code here -------- #
        # Compute non-normal(historical) VaR and CVaR
        VaRout[portN, q] = Loss_port[int(math.ceil(Nout * alf))-1]
        CVaRout[portN, q] = (1 / (Nout * (1-alf))) * ((math.ceil(Nout * alf) - Nout * alf) * VaRout[portN,q]+ sum(Loss_port[int(math.ceil(Nout * alf)):]))
        # Compute normal VaR and CVaR
        VaRinN[portN, q] = np.mean(Loss_port) + scs.norm.ppf(alf) * np.std(Loss_port)        
        CVaRinN[portN, q] = np.mean(Loss_port) + (scs.norm.pdf(scs.norm.ppf(alf)) / (1-alf)) * np.std(Loss_port)

        
# Perform 100 trials
N_trials = 100

VaRinMC1 = {}
VaRinMC2 = {}
VaRinN1 = {}
VaRinN2 = {}
CVaRinMC1 = {}
CVaRinMC2 = {}
CVaRinN1 = {}
CVaRinN2 = {}

startTime = time.time() #start time for stimulation

for portN in range(2):
    for q in range(alphas.size):
        VaRinMC1[portN, q] = np.zeros(N_trials)
        VaRinMC2[portN, q] = np.zeros(N_trials)
        VaRinN1[portN, q] = np.zeros(N_trials)
        VaRinN2[portN, q] = np.zeros(N_trials)
        CVaRinMC1[portN, q] = np.zeros(N_trials)
        CVaRinMC2[portN, q] = np.zeros(N_trials)
        CVaRinN1[portN, q] = np.zeros(N_trials)
        CVaRinN2[portN, q] = np.zeros(N_trials)


for tr in range(1, N_trials + 1):
    # Monte Carlo approximation 1

    # -------- Insert your code here -------- #
    # to store the losses
    Losses_inMC1= np.zeros((Nin, K))# for 5000 scenarios and 100 counter parties
    Nss_MC1 = np.int(Nin / Ns) #number of systemic scenarios
    # create y to store systematic risk components
    y_MC1 = np.zeros((Nss_MC1, NumRF)) # for 1000 scenarios and 50 credit drivers
    

    for s in range(1, np.int(np.ceil(Nin / Ns) + 1)): # 1000 systemic scenarios
        # -------- Insert your code here -------- #
        random_MC1 = np.random.normal(0,1, (NumRF, 1)) #from N(0,1) 
        #generate correlated Normal random numbers using Cholesky decomp of rho
        y_MC1[s-1,:] = np.dot(sqrt_rho,random_MC1).T 

        for si in range(1, Ns + 1): # 5 idiosyncratic scenarios for each systemic
            # -------- Insert your code here -------- #
            # define idiosyncratic component for 100 counter parties from N(0,1)
            z_MC1 = np.random.normal(0, 1, (K,1))
            for cp in range(1, K+1): 
               RF = int(driver[cp-1]) #get credit driver index for that counter party
               #Calculate creditworthiness index
               w_MC1 = beta[cp-1] * y_MC1[s-1, RF-1] + np.sqrt(1 - beta[cp-1]**2) * z_MC1[cp-1]
               # append w to CS boundaries, and sort the values
               wMC1_CS_Bdry= np.sort(np.append(w_MC1, CS_Bdry[cp-1,:]))
               #find credit state index
               CS_MC1_index=np.argwhere(wMC1_CS_Bdry == w_MC1)
               # Calculate losses for MC1 approximation (5000 x 100)
               Losses_inMC1[5*(s-1)+si-1,cp-1] = exposure[cp-1, CS_MC1_index] 

    

    # Monte Carlo approximation 2

    # -------- Insert your code here -------- #
    # to store the losses
    Losses_inMC2= np.zeros((Nin, K))# for 5000 scenarios and 100 counter parties
    # create y to store systematic risk components
    y_MC2 = np.zeros((Nin, NumRF)) # for 1000 scenarios and 50 credit drivers
    # define idiosyncratic component for 100 counter parties from N(0,1)
    z_MC2 = np.random.normal(0, 1, (K,1))

    for s in range(1, Nin + 1): # systemic scenarios (1 idiosyncratic scenario for each systemic)
        # -------- Insert your code here -------- #
        random_MC2 = np.random.normal(0,1, (NumRF, 1)) #from N(0,1) 
        #generate correlated Normal random numbers using Cholesky decomp of rho
        y_MC2[s-1,:] = np.dot(sqrt_rho,random_MC2).T 
        for cp in range(1, K+1): 
            RF = int(driver[cp-1]) #get credit driver index for that counter party
            #Calculate creditworthiness index
            w_MC2 = beta[cp-1] * y_MC2[s-1, RF-1] + np.sqrt(1 - beta[cp-1]**2) * z_MC2[cp-1]
            # append w to CS boundaries, and sort the values
            wMC2_CS_Bdry= np.sort(np.append(w_MC2, CS_Bdry[cp-1,:]))
            #find credit state index
            CS_MC2_index=np.argwhere(wMC2_CS_Bdry == w_MC2)
            # Calculate losses for MC2 approximation (5000 x 100)
            Losses_inMC2[s-1,cp-1] = exposure[cp-1, CS_MC2_index] 
        

    # Compute VaR and CVaR for portfolio 1 and 2

    for portN in range(2): 
        for q in range(alphas.size):
            alf = alphas[q]
            # -------- Insert your code here -------- #
            # Compute portfolio loss
            portf_loss_inMC1 = np.sort(np.dot(Losses_inMC1,x0[portN]))
            portf_loss_inMC2 = np.sort(np.dot(Losses_inMC2,x0[portN]))
            mu_MC1 = np.mean(Losses_inMC1, axis=0).reshape((K))
            var_MC1 = np.cov(Losses_inMC1, rowvar=False)
            mu_MC2 = np.mean(Losses_inMC2, axis=0).reshape((K))
            var_MC2 = np.cov(Losses_inMC2, rowvar=False)
            # Compute portfolio mean loss mu_p_MC1 and portfolio standard deviation of losses sigma_p_MC1
            mu_p_MC1 = np.mean(portf_loss_inMC1)
            sigma_p_MC1 = np.std(portf_loss_inMC1)
            # Compute portfolio mean loss mu_p_MC2 and portfolio standard deviation of losses sigma_p_MC2
            mu_p_MC2 = np.mean(portf_loss_inMC2)
            sigma_p_MC2 = np.std(portf_loss_inMC2)
            
            # Compute VaR and CVaR for the current trial
            VaRinMC1[portN, q][tr - 1] = portf_loss_inMC1[int(math.ceil(Nin * alf))-1]
            VaRinMC2[portN, q][tr - 1] = portf_loss_inMC2[int(math.ceil(Nin * alf))-1]
            VaRinN1[portN, q][tr - 1] =  mu_p_MC1 + scs.norm.ppf(alf) * sigma_p_MC1
            VaRinN2[portN, q][tr - 1] =  mu_p_MC2 + scs.norm.ppf(alf) * sigma_p_MC2
            
            CVaRinMC1[portN, q][tr - 1] = (1 / (Nin * (1-alf))) * ((math.ceil(Nin * alf) - Nin * alf) * VaRinMC1[portN,q][tr - 1]+ sum(portf_loss_inMC1[int(math.ceil(Nin * alf)):]))
            CVaRinMC2[portN, q][tr - 1] = (1 / (Nin * (1-alf))) * ((math.ceil(Nin * alf) - Nin * alf) * VaRinMC2[portN,q][tr - 1]+ sum(portf_loss_inMC2[int(math.ceil(Nin * alf)):]))
            CVaRinN1[portN, q][tr - 1] =  mu_p_MC1 + (scs.norm.pdf(scs.norm.ppf(alf)) / (1-alf)) * sigma_p_MC1
            CVaRinN2[portN, q][tr - 1] =  mu_p_MC2 + (scs.norm.pdf(scs.norm.ppf(alf)) / (1-alf)) * sigma_p_MC2

executionTime = (time.time() - startTime) #check run time for stimulations
print('Execution time in seconds: ' + str(executionTime))

# Display VaR and CVaR

for portN in range(2):
    print('\nPortfolio {}:\n'.format(portN + 1))
    for q in range(alphas.size):
        alf = alphas[q]
        print('Out-of-sample: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f' % (
        100 * alf, VaRout[portN, q], 100 * alf, CVaRout[portN, q]))
        print('In-sample MC1: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f' % (
        100 * alf, np.mean(VaRinMC1[portN, q]), 100 * alf, np.mean(CVaRinMC1[portN, q])))
        print('In-sample MC2: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f' % (
        100 * alf, np.mean(VaRinMC2[portN, q]), 100 * alf, np.mean(CVaRinMC2[portN, q])))
        print('In-sample No: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f' % (
        100 * alf, VaRinN[portN, q], 100 * alf, CVaRinN[portN, q]))
        print('In-sample N1: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f' % (
        100 * alf, np.mean(VaRinN1[portN, q]), 100 * alf, np.mean(CVaRinN1[portN, q])))
        print('In-sample N2: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n' % (
        100 * alf, np.mean(VaRinN2[portN, q]), 100 * alf, np.mean(CVaRinN2[portN, q])))

# Plot results
# Figure (1): Non-normal 100000 v.s. Normal 100000 of portfolio 1
plt.figure(figsize=(20,10),dpi=900)
Loss_port1=np.sort(np.dot(Losses_out.toarray(),x0[0]))
frequencyCounts, binLocations, patches = plt.hist(Loss_port1, 100)
normf = (1 / (np.std(Loss_port1) * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((binLocations - np.mean(Loss_port1)) / np.std(Loss_port1)) ** 2)
normf = normf * sum(frequencyCounts) / sum(normf)
plt.plot(binLocations, normf, color='r', linewidth=2.0)
plt.plot([VaRinN[0, 0], VaRinN[0, 0]], [0, max(frequencyCounts) / 1.9], color='r', linewidth=1, linestyle='-.')
plt.text(0.9 * VaRinN[0, 0], max(frequencyCounts) / 1.9, 'VaRinN\n(99.0%)')
plt.plot([CVaRinN[0, 0], CVaRinN[0, 0]], [0, max(frequencyCounts) / 2.4], color='r', linewidth=1, linestyle='-.')
plt.text(0.9 * CVaRinN[0, 0], max(frequencyCounts) / 2.4, 'CVaRinN\n(99.0%)')
plt.plot([VaRinN[0, 1], VaRinN[0, 1]], [0, max(frequencyCounts) / 1.3], color='r', linewidth=1, linestyle='-.')
plt.text(0.91 * VaRinN[0, 1], max(frequencyCounts) / 1.3, 'VaRinN\n(99.9%)')
plt.plot([CVaRinN[0, 1], CVaRinN[0, 1]], [0, max(frequencyCounts) / 1.1], color='r', linewidth=1, linestyle='-.')
plt.text(0.93 * CVaRinN[0, 1], max(frequencyCounts) / 1.1, 'CVaRinN\n(99.9%)')

plt.plot([VaRout[0, 0], VaRout[0, 0]], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.92 * VaRout[0, 0], max(frequencyCounts) / 1.6, 'VaRout\n(99.0%)')
plt.plot([CVaRout[0, 0], CVaRout[0, 0]], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * CVaRout[0, 0], max(frequencyCounts) / 1.6, 'CVaRout\n(99.0%)')
plt.plot([VaRout[0, 1], VaRout[0, 1]], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * VaRout[0, 1], max(frequencyCounts) / 1.6, 'VaRout\n(99.9%)')
plt.plot([CVaRout[0, 1], CVaRout[0, 1]], [0, max(frequencyCounts) / 1.3], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * CVaRout[0, 1], max(frequencyCounts) / 1.3, 'CVaRout\n(99.9%)')

plt.xlabel('1-day loss in $ value on portfolio 1')
plt.ylabel('Frequency')
plt.title('True Distribution on Portfolio 1')
plt.draw()

# Figure (2): Non-normal 100000 v.s. Normal 100000 of portfolio 2
plt.figure(figsize=(20,10),dpi=900)
Loss_port2=np.sort(np.dot(Losses_out.toarray(),x0[1]))
frequencyCounts, binLocations, patches = plt.hist(Loss_port2, 100)
normf = (1 / (np.std(Loss_port2) * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((binLocations - np.mean(Loss_port2)) / np.std(Loss_port2)) ** 2)
normf = normf * sum(frequencyCounts) / sum(normf)
plt.plot(binLocations, normf, color='r', linewidth=2.0)
plt.plot([VaRinN[1, 0], VaRinN[1, 0]], [0, max(frequencyCounts) / 1.9], color='r', linewidth=1, linestyle='-.')
plt.text(0.9 * VaRinN[1, 0], max(frequencyCounts) / 1.9, 'VaRinN\n(99.0%)')
plt.plot([CVaRinN[1, 0], CVaRinN[1, 0]], [0, max(frequencyCounts) / 2.4], color='r', linewidth=1, linestyle='-.')
plt.text(0.9 * CVaRinN[0, 0], max(frequencyCounts) / 2.4, 'CVaRinN\n(99.0%)')
plt.plot([VaRinN[1, 1], VaRinN[1, 1]], [0, max(frequencyCounts) / 1.3], color='r', linewidth=1, linestyle='-.')
plt.text(0.91 * VaRinN[0, 1], max(frequencyCounts) / 1.3, 'VaRinN\n(99.9%)')
plt.plot([CVaRinN[1, 1], CVaRinN[1, 1]], [0, max(frequencyCounts) / 1.1], color='r', linewidth=1, linestyle='-.')
plt.text(0.93 * CVaRinN[0, 1], max(frequencyCounts) / 1.1, 'CVaRinN\n(99.9%)')

plt.plot([VaRout[1, 0], VaRout[1, 0]], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.92 * VaRout[1, 0], max(frequencyCounts) / 1.6, 'VaRout\n(99.0%)')
plt.plot([CVaRout[1, 0], CVaRout[1, 0]], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * CVaRout[1, 0], max(frequencyCounts) / 1.6, 'CVaRout\n(99.0%)')
plt.plot([VaRout[1, 1], VaRout[1 ,1]], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * VaRout[1, 1], max(frequencyCounts) / 1.6, 'VaRout\n(99.9%)')
plt.plot([CVaRout[1, 1], CVaRout[1, 1]], [0, max(frequencyCounts) / 1.3], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * CVaRout[1, 1], max(frequencyCounts) / 1.3, 'CVaRout\n(99.9%)')

plt.xlabel('1-day loss in $ value on portfolio 2')
plt.ylabel('Frequency')
plt.title('True Distribution on Portfolio 2')
plt.draw()

# Figure (3): Non-normal 1000x5 v.s. Normal 1000x5 of portfolio 1
plt.figure(figsize=(20,10),dpi=900)
portf_loss1_inMC1 = np.sort(np.dot(Losses_inMC1,x0[0]))
frequencyCounts, binLocations, patches = plt.hist(portf_loss1_inMC1, 100)
normf = (1 / (np.std(portf_loss1_inMC1) * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((binLocations - np.mean(portf_loss1_inMC1)) / np.std(portf_loss1_inMC1)) ** 2)
normf = normf * sum(frequencyCounts) / sum(normf)
plt.plot(binLocations, normf, color='r', linewidth=2.0)
plt.plot([np.mean(VaRinN1[0, 0]), np.mean(VaRinN1[0, 0])], [0, max(frequencyCounts) / 1.9], color='r', linewidth=1, linestyle='-.')
plt.text(0.9 * np.mean(VaRinN1[0, 0]), max(frequencyCounts) / 1.9, 'VaRinN1\n(99.0%)')
plt.plot([np.mean(CVaRinN1[0, 0]), np.mean(CVaRinN1[0, 0])], [0, max(frequencyCounts) / 2.4], color='r', linewidth=1, linestyle='-.')
plt.text(0.95 * np.mean(CVaRinN1[0, 0]), max(frequencyCounts) / 2.4, 'CVaRinN1\n(99.0%)')
plt.plot([np.mean(VaRinN1[0, 1]), np.mean(VaRinN1[0, 1])], [0, max(frequencyCounts) / 1.3], color='r', linewidth=1, linestyle='-.')
plt.text(0.91 * np.mean(VaRinN1[0, 1]), max(frequencyCounts) / 1.3, 'VaRinN1\n(99.9%)')
plt.plot([np.mean(CVaRinN1[0, 1]), np.mean(CVaRinN1[0, 1])], [0, max(frequencyCounts) / 1.1], color='r', linewidth=1, linestyle='-.')
plt.text(0.93 * np.mean(CVaRinN1[0, 1]), max(frequencyCounts) / 1.1, 'CVaRinN1\n(99.9%)')

plt.plot([np.mean(VaRinMC1[0, 0]), np.mean(VaRinMC1[0, 0])], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * np.mean(VaRinMC1[0, 0]), max(frequencyCounts) / 1.6, 'VaRinMC1\n(99.0%)')
plt.plot([np.mean(CVaRinMC1[0, 0]), np.mean(CVaRinMC1[0, 0])], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * np.mean(CVaRinMC1[0, 0]), max(frequencyCounts) / 1.6, 'CVaRinMC1\n(99.0%)')
plt.plot([np.mean(VaRinMC1[0, 1]), np.mean(VaRinMC1[0, 1])], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * np.mean(VaRinMC1[0, 1]), max(frequencyCounts) / 1.6, 'VaRinMC1\n(99.9%)')
plt.plot([np.mean(CVaRinMC1[0, 1]), np.mean(CVaRinMC1[0, 1])], [0, max(frequencyCounts) / 1.3], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * np.mean(CVaRinMC1[0, 1]), max(frequencyCounts) / 1.3, 'CVaRinMC1\n(99.9%)')

plt.xlabel('1-day loss in $ value on portfolio 1')
plt.ylabel('Frequency')
plt.title('MC1 on Portfolio 1')
plt.draw()


# Figure (4): Non-normal 1000x5 v.s. Normal 1000x5 of portfolio 2
plt.figure(figsize=(20,10),dpi=900)
portf_loss2_inMC1 = np.sort(np.dot(Losses_inMC1,x0[1]))
frequencyCounts, binLocations, patches = plt.hist(portf_loss2_inMC1, 100)
normf = (1 / (np.std(portf_loss2_inMC1) * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((binLocations - np.mean(portf_loss2_inMC1)) / np.std(portf_loss2_inMC1)) ** 2)
normf = normf * sum(frequencyCounts) / sum(normf)
plt.plot(binLocations, normf, color='r', linewidth=2.0)
plt.plot([np.mean(VaRinN1[1, 0]), np.mean(VaRinN1[1, 0])], [0, max(frequencyCounts) / 1.9], color='r', linewidth=1, linestyle='-.')
plt.text(0.9 * np.mean(VaRinN1[1, 0]), max(frequencyCounts) / 1.9, 'VaRinN1\n(99.0%)')
plt.plot([np.mean(CVaRinN1[1, 0]), np.mean(CVaRinN1[1, 0])], [0, max(frequencyCounts) / 2.4], color='r', linewidth=1, linestyle='-.')
plt.text(0.95 * np.mean(CVaRinN1[1, 0]), max(frequencyCounts) / 2.4, 'CVaRinN1\n(99.0%)')
plt.plot([np.mean(VaRinN1[1, 1]), np.mean(VaRinN1[1, 1])], [0, max(frequencyCounts) / 1.3], color='r', linewidth=1, linestyle='-.')
plt.text(0.91 * np.mean(VaRinN1[1, 1]), max(frequencyCounts) / 1.3, 'VaRinN1\n(99.9%)')
plt.plot([np.mean(CVaRinN1[1, 1]), np.mean(CVaRinN1[1, 1])], [0, max(frequencyCounts) / 1.1], color='r', linewidth=1, linestyle='-.')
plt.text(0.93 * np.mean(CVaRinN1[1, 1]), max(frequencyCounts) / 1.1, 'CVaRinN1\n(99.9%)')

plt.plot([np.mean(VaRinMC1[1, 0]), np.mean(VaRinMC1[1, 0])], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * np.mean(VaRinMC1[1, 0]), max(frequencyCounts) / 1.6, 'VaRinMC1\n(99.0%)')
plt.plot([np.mean(CVaRinMC1[1, 0]), np.mean(CVaRinMC1[1, 0])], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * np.mean(CVaRinMC1[1, 0]), max(frequencyCounts) / 1.6, 'CVaRinMC1\n(99.0%)')
plt.plot([np.mean(VaRinMC1[1, 1]), np.mean(VaRinMC1[1, 1])], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * np.mean(VaRinMC1[1, 1]), max(frequencyCounts) / 1.6, 'VaRinMC1\n(99.9%)')
plt.plot([np.mean(CVaRinMC1[1, 1]), np.mean(CVaRinMC1[1, 1])], [0, max(frequencyCounts) / 1.3], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * np.mean(CVaRinMC1[1, 1]), max(frequencyCounts) / 1.3, 'CVaRinMC1\n(99.9%)')

plt.xlabel('1-day loss in $ value on portfolio 2')
plt.ylabel('Frequency')
plt.title('MC1 on Portfolio 2')
plt.draw()


# Figure (5): Non-normal 5000 v.s. Normal 5000 of portfolio 1
plt.figure(figsize=(20,10),dpi=900)
portf_loss1_inMC2 = np.sort(np.dot(Losses_inMC2,x0[0]))
frequencyCounts, binLocations, patches = plt.hist(portf_loss1_inMC2, 100)
normf = (1 / (np.std(portf_loss1_inMC2) * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((binLocations - np.mean(portf_loss1_inMC2)) / np.std(portf_loss1_inMC2)) ** 2)
normf = normf * sum(frequencyCounts) / sum(normf)
plt.plot(binLocations, normf, color='r', linewidth=2.0)
plt.plot([np.mean(VaRinN2[0, 0]), np.mean(VaRinN2[0, 0])], [0, max(frequencyCounts) / 1.9], color='r', linewidth=1, linestyle='-.')
plt.text(0.9 * np.mean(VaRinN1[0, 0]), max(frequencyCounts) / 1.9, 'VaRinN2\n(99.0%)')
plt.plot([np.mean(CVaRinN2[0, 0]), np.mean(CVaRinN2[0, 0])], [0, max(frequencyCounts) / 2.4], color='r', linewidth=1, linestyle='-.')
plt.text(0.95 * np.mean(CVaRinN2[0, 0]), max(frequencyCounts) / 2.4, 'CVaRinN2\n(99.0%)')
plt.plot([np.mean(VaRinN2[0, 1]), np.mean(VaRinN2[0, 1])], [0, max(frequencyCounts) / 1.3], color='r', linewidth=1, linestyle='-.')
plt.text(0.91 * np.mean(VaRinN2[0, 1]), max(frequencyCounts) / 1.3, 'VaRinN2\n(99.9%)')
plt.plot([np.mean(CVaRinN2[0, 1]), np.mean(CVaRinN2[0, 1])], [0, max(frequencyCounts) / 1.1], color='r', linewidth=1, linestyle='-.')
plt.text(0.93 * np.mean(CVaRinN2[0, 1]), max(frequencyCounts) / 1.1, 'CVaRinN2\n(99.9%)')

plt.plot([np.mean(VaRinMC2[0, 0]), np.mean(VaRinMC2[0, 0])], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * np.mean(VaRinMC2[0, 0]), max(frequencyCounts) / 1.6, 'VaRinMC2\n(99.0%)')
plt.plot([np.mean(CVaRinMC2[0, 0]), np.mean(CVaRinMC2[0, 0])], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * np.mean(CVaRinMC2[0, 0]), max(frequencyCounts) / 1.6, 'CVaRinMC2\n(99.0%)')
plt.plot([np.mean(VaRinMC2[0, 1]), np.mean(VaRinMC2[0, 1])], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * np.mean(VaRinMC2[0, 1]), max(frequencyCounts) / 1.6, 'VaRinMC2\n(99.9%)')
plt.plot([np.mean(CVaRinMC2[0, 1]), np.mean(CVaRinMC2[0, 1])], [0, max(frequencyCounts) / 1.3], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * np.mean(CVaRinMC2[0, 1]), max(frequencyCounts) / 1.3, 'CVaRinMC2\n(99.9%)')

plt.xlabel('1-day loss in $ value on portfolio 1')
plt.ylabel('Frequency')
plt.title('MC2 on Portfolio 1')
plt.draw()


# Figure (6): Non-normal 5000 v.s. Normal 5000 of portfolio 2
plt.figure(figsize=(20,10),dpi=900)
portf_loss2_inMC2 = np.sort(np.dot(Losses_inMC2,x0[1]))
frequencyCounts, binLocations, patches = plt.hist(portf_loss2_inMC2, 100)
normf = (1 / (np.std(portf_loss2_inMC2) * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((binLocations - np.mean(portf_loss2_inMC2)) / np.std(portf_loss2_inMC2)) ** 2)
normf = normf * sum(frequencyCounts) / sum(normf)
plt.plot(binLocations, normf, color='r', linewidth=2.0)
plt.plot([np.mean(VaRinN2[1, 0]), np.mean(VaRinN2[1, 0])], [0, max(frequencyCounts) / 1.9], color='r', linewidth=1, linestyle='-.')
plt.text(0.9 * np.mean(VaRinN1[1, 0]), max(frequencyCounts) / 1.9, 'VaRinN2\n(99.0%)')
plt.plot([np.mean(CVaRinN2[1, 0]), np.mean(CVaRinN2[1, 0])], [0, max(frequencyCounts) / 2.4], color='r', linewidth=1, linestyle='-.')
plt.text(0.95 * np.mean(CVaRinN2[1, 0]), max(frequencyCounts) / 2.4, 'CVaRinN2\n(99.0%)')
plt.plot([np.mean(VaRinN2[1, 1]), np.mean(VaRinN2[1, 1])], [0, max(frequencyCounts) / 1.3], color='r', linewidth=1, linestyle='-.')
plt.text(0.91 * np.mean(VaRinN2[1, 1]), max(frequencyCounts) / 1.3, 'VaRinN2\n(99.9%)')
plt.plot([np.mean(CVaRinN2[1, 1]), np.mean(CVaRinN2[1, 1])], [0, max(frequencyCounts) / 1.1], color='r', linewidth=1, linestyle='-.')
plt.text(0.93 * np.mean(CVaRinN2[1, 1]), max(frequencyCounts) / 1.1, 'CVaRinN2\n(99.9%)')

plt.plot([np.mean(VaRinMC2[1, 0]), np.mean(VaRinMC2[1, 0])], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * np.mean(VaRinMC2[1, 0]), max(frequencyCounts) / 1.6, 'VaRinMC2\n(99.0%)')
plt.plot([np.mean(CVaRinMC2[1, 0]), np.mean(CVaRinMC2[1, 0])], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * np.mean(CVaRinMC2[1, 0]), max(frequencyCounts) / 1.6, 'CVaRinMC2\n(99.0%)')
plt.plot([np.mean(VaRinMC2[1, 1]), np.mean(VaRinMC2[1, 1])], [0, max(frequencyCounts) / 1.6], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * np.mean(VaRinMC2[1, 1]), max(frequencyCounts) / 1.6, 'VaRinMC2\n(99.9%)')
plt.plot([np.mean(CVaRinMC2[1, 1]), np.mean(CVaRinMC2[1, 1])], [0, max(frequencyCounts) / 1.3], color='g', linewidth=1, linestyle='-.')
plt.text(0.97 * np.mean(CVaRinMC2[1, 1]), max(frequencyCounts) / 1.3, 'CVaRinMC2\n(99.9%)')

plt.xlabel('1-day loss in $ value on portfolio 2')
plt.ylabel('Frequency')
plt.title('MC2 on Portfolio 2')
plt.draw()
