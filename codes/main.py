#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 01 17:51:39 2023

@author: Ali Moghaddam
"""


import numpy as np
import matplotlib.pyplot as plt
from utils import *
import numpy as np

def probability_to_var(prob,num):
    return np.dot(prob,np.arange(num+1)**2) - np.dot(prob,np.arange(num+1))**2.

def spin_prob(num,rho):

    """
    Calculate the probability of the number of spins up in subsystem from the reduced density matrix.

    Parameters
    ----------
    num : int
        Number of spins in the subsystem.
    rho : array
        Diagonal elements of the reduced density matrix.

    Returns
    -------
    sz_prob : array
        Probabilities of the number of spins up in the subsystem.
    """

    sz_prob = np.zeros(num+1)
    for i in range(num+1):
        for j in range(2**num):
            if bin(j)[2:].count('1') == i:
                sz_prob[i] += rho[j].real
    return sz_prob

def run_ref(circuit_length, circuit_depth, phi_cte, gate_randomness, measurement_rate, measure_strength, PBC):


    initial_state = Neel(circuit_length)
    state = QuantumTrajectoryDynamics(circuit_length, initial_state, 0, {}, {}, {})
    dic_var = [0.]

    for _ in range(circuit_depth):

        state.even_time_unitary()
        state.timestep += 1

        state.measurement_layer(measurement_rate, measure_strength)
        state.timestep += 1



        rho_final = reduced_density_matrix_from_state(state.psi, [circuit_length//2,circuit_length - circuit_length//2])
        sz_prob = spin_prob(circuit_length//2,np.diagonal(rho_final))
        dic_var.append(probability_to_var(sz_prob,circuit_length//2))


        state.odd_time_unitary()
        state.timestep += 1

        state.measurement_layer(measurement_rate, measure_strength)
        state.timestep += 1

        rho_final = reduced_density_matrix_from_state(state.psi, [circuit_length//2,circuit_length - circuit_length//2])
        sz_prob = spin_prob(circuit_length//2,np.diagonal(rho_final))
        dic_var.append(probability_to_var(sz_prob,circuit_length//2))

    # let's return the outcomes 

    measurement_outcomes_dict = state.outcomes


    return np.array(dic_var).reshape(1,-1), measurement_outcomes_dict



# num_realizations=200
p =0.40
length = 10
depth = 20

variance, outcome_dict = run_ref(length, depth, 0., 1., p, 1., True)
outcome_data_array = 2 * np.ones((length,2 * depth))

for key in outcome_dict.keys():
    if key[1] % 2 == 1: # odd time step
        outcome_data_array[key[0],key[1]//2] = outcome_dict[key]


print(outcome_dict)
print(outcome_data_array[:,0])

print(outcome_data_array[:,3])



# var_time_series = run_ref(length, depth, 0., 1., p, 1., True)

# for _ in range(num_realizations):
#     var_different_realizatins = np.concatenate((run_ref(length, depth, 0., 1., p, 1., True),var_time_series))

# var_avg = var_different_realizatins.mean(axis=0)
# var_std = var_different_realizatins.std(axis=0)


# fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

# axs[0].plot(var_avg[::2])
# axs[0].fill_between(np.arange(len(var_avg[::2])), var_avg[::2] - var_std[::2], var_avg[::2] + var_std[::2], alpha=0.5)

# axs[1].plot(var_avg[1::2])
# axs[1].fill_between(np.arange(len(var_avg[1::2])), var_avg[1::2] - var_std[1::2], var_avg[1::2] + var_std[1::2], alpha=0.5)

# axs[0].set_title('Even time steps'+f'var={var_avg[40::2].mean()}')
# axs[1].set_title('Odd time steps'+f'var={var_avg[41::2].mean()}')

# var_avg[40::2].mean(),var_avg[41::2].mean()

# print(var_avg[40::2].mean(),var_avg[41::2].mean())

# plt.savefig(f'var_time_series-p={p}-L={length}.png')








