#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Apr 13 11:51:39 2023

@author: Ali Moghaddam
"""


import numpy as np
from scipy.stats import unitary_group as randU
from scipy.sparse import csc_matrix, csr_matrix, spmatrix, bsr_matrix

# import scipy.sparse as sp
# import scipy.linalg as la, scipy.sparse as sp
# import pylab as py
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# import time
# from datetime import datetime
# np.random.seed(14)
from util_func import *

twoqubit_U = twoqubit_unitary_U1
Neel = Neel_state_direct

sigma_x = np.array([[0.,1.],[1.,0.]],dtype=complex)


class QuantumTrajectoryDynamics:
    
    """ 
        Create and evolves a new manybody state of N qubits.  N := "num_qubits" or "circuit length"
        Evolution consits of (a) two-qubit unitary gates which are successively applied to neighboring qubits.
                         and (b) projective measurements of single qubits

    Args:
    
        num_qubits: An integer indicating the number of qubits or in other words the length of quantum circuit
        psi : manybody state of N qubits. A 2^N-dimensional numpy array.
        timestep: An integer indicating the number of time steps in the evolution.

        outcomes: A 2D array containing the outcomes of single-qubit measurements with 0th axis corresponding to position and 1st axis coresponding to timestep.
        
        probabilities: A dictionary containing the probabilities of outcome |0> for single-qubit measurements.

    """

    def __init__(self, num_qubits: int, psi: np.ndarray, timestep: int, outcomes: dict, probabilities: dict, fixed_outcomes: dict):
        self.num_qubits = num_qubits
        self.psi = psi
        self.timestep = timestep
        self.outcomes = outcomes
        self.probabilities = probabilities
        self.fixed_outcomes = fixed_outcomes

    def even_time_unitary(self) -> np.ndarray:
        """ Applies two-qubit unitary gates successively to neighboring qubits in even time steps.

        Args:
            phi_cte: A constant (6-component) set of unitary phases. Default all 0.
            randomness: A float indicating the amount of randomness in unitary phases. Default is 1.

        Returns:
            psi: The updated manybody state of N qubits. A 2^N-dimensional numpy array.

        """

        num_TwoQubitGate = self.num_qubits//2
        
        for m in range(num_TwoQubitGate):
            self.psi = two_qubit_gate_on_circuit(self.psi, randU.rvs(4), 2 * m)       # This function applies two-qubit gate on circuit
        


    def odd_time_unitary(self, PBC = True) -> np.ndarray:
        """ Applies two-qubit unitary gates successively to neighboring qubits in odd time steps.

        Args:
            phi_cte: A constant (6-component) set of unitary phases. Default all 0.
            randomness: A float indicating the amount of randomness in unitary phases. Default is 1.
            PBC: A boolean indicating if we have periodic boundary condiction (PBC) or not.

        Returns:
            psi: The updated manybody state of N qubits. A 2^N-dimensional numpy array.

        """
        
        if PBC:        
            self.psi = self.psi.reshape(2 , 2**(self.num_qubits-1)).transpose(1, 0)
            self.even_time_unitary()
            self.psi = self.psi.reshape(2**(self.num_qubits-1), 2).transpose(1, 0)
        
        else:
            num_TwoQubitGate = self.num_qubits//2 - 1    # The number of gates for OBC case.

            for m in range(num_TwoQubitGate):
                self.psi = two_qubit_gate_on_circuit(self.psi, randU.rvs(4), 2 * m + 1)



    def measurement_layer(self, measurement_rate: float = 0., measure_strength: float = 1.) -> np.ndarray:
        """ Applies projective measurements of single qubits.

        Args:
            measurement_rate: A float indicating the probability of measuring each qubit. Default is 0. for no measurement at all. 

        Returns:
            psi: The updated manybody state of N qubits. A 2^N-dimensional numpy array.

        """
        
        for m in range(self.num_qubits):
            if np.random.rand() < measurement_rate:
                self.psi, prob_0, measurement_outcome = measure_weak_single_qubit_sz(self.psi, m, measure_strength)
                self.outcomes[(m, self.timestep)] = measurement_outcome
                self.probabilities[(m, self.timestep)] = prob_0    


    def steering_layer(self) -> np.ndarray:
        """ 
        Applies local single-qubit corrections to the manybody state of N qubits 
        to match with the desired outcomes of single-qubit measurements.
       
        Returns:
            psi: The updated manybody state of N qubits. A 2^N-dimensional numpy array.

        """
        
        for measured in [key for key in self.outcomes.keys() if key[1] == self.timestep]:
            if self.outcomes[measured] != self.fixed_outcomes[measured]:
                m = measured[0]
                self.psi = self.psi.reshape(2**m , 2, -1).transpose(1, 0, 2).reshape(2, -1)
                self.psi = np.matmul(sigma_x, self.psi)
                self.psi = self.psi.reshape(2, 2**m , -1).transpose(1, 0, 2)

    
                


    def subsys_entanglemententropy_and_spinvariance(self, partitions: list, order: float = 1) -> float:
        """ Calculates the entanglement entropy and spin variance for a given subsytem in a bipartitioning.
            Important note: the two subsystems can be decomposed of disjoint smaler regions (encoded in the "partitions" list).
            For instance if we have an overall of 16 qubits, then partitions = [8,8] or [4,12] is simple bipartitioning,
            but we can have also partitions = [4,4,4,4] or even [2,4,2,3,1,4].
            By definition subsystem A and B corespond to the even-index and odd-index elements of the list "partitions", respectively.
            Obviously, the total number of qubits in the system must be equal to the sum of the elements of the list "partitions".


        Args:
            order: An float indicating the type of entanglement entropy. Default is 1 corresponding to 'von_Neumann'.

        Returns:
            S: The entanglement entropy of the subsystem for a the manybody state.
            varSz: The spin variance of the subsystem for a the manybody state.

        """
        
        rho = reduced_density_matrix_from_state(self.psi, partitions, subsys='A')

        # Calculate the entanglement entropy: 

        eigvals_rho = np.linalg.eigvalsh(rho)
        non_zero_eigvals_rho = [l for l in eigvals_rho if l > 1e-10]

        if order == 1: # von Neumann entropy S = -Tr(rho * log(rho)) = -sum(l_i * log(l_i))
            S = -np.sum([l * np.log(l) for l in non_zero_eigvals_rho])
        elif order > 1: # Renyi entropy of order = order S = -1/(1-order) * Tr(log(rho^order)) = -1/(1-order) * log(sum(l_i^order))
            S =  -np.sum([np.log(l**order) for l in non_zero_eigvals_rho])/(1 - order)
        else:
            raise ValueError("Order of entanglement entropy must be greater than 0")

        if abs(S) < 1e-10:
            S = 0.

        # Calculate the spin variance:

        num_qubits_subsys_A = sum(partitions[::2])
        Sz_tot = Sz_tot_diagon(num_qubits_subsys_A).flatten()
        rho_diag = np.diagonal(rho).flatten()
        varSz = np.dot(rho_diag, Sz_tot**2.) - np.dot(rho_diag, Sz_tot)**2. # varSz = <Sz^2> - <Sz>^2

        return S, varSz
    
    def full_circuit_observables(self):
        Sz_total = bin(abs(self.psi.flatten()).argmax() )[2:].count('1') #- self.num_qubits//2
        return Sz_total
    

    def total_sys_rho(self):
        psi = self.psi.reshape(2**self.num_qubits,1)
        psi_sparse = csc_matrix(psi,dtype=complex)
        # rho_tot_sparse = psi_sparse * psi_sparse.transpose().conj()
        rho_tot_test = psi * np.conjugate(psi.T)
        return rho_tot_test
        
        




class ConservedTrajectoryDynamics(QuantumTrajectoryDynamics):

    """ A subclass of class "QuantumTrajectoryDynamics" for simulating quantum dynamics under U(1) conservation meaning
    that with two-qubit gates which keep the total charge of two qubtits intact.
    """ 

    def __init__(self, num_qubits: int, psi: np.ndarray, timestep: int, outcomes: dict, probabilities: dict, fixed_outcomes: dict):
        super().__init__(num_qubits, psi, timestep, outcomes, probabilities, fixed_outcomes)


    def even_time_unitary(self, phi_cte = np.zeros((6,1)), randomness = 1) -> np.ndarray:
        """ Applies two-qubit U(1) charge-conserving unitary gates successively to neighboring qubits in even time steps.

        Args:
            phi_cte: A constant (6-component) set of unitary phases. Default all 0.
            randomness: A float indicating the amount of randomness in unitary phases. Default is 1.

        Returns:
            psi: The updated manybody state of N qubits. A 2^N-dimensional numpy array.

        """

        num_TwoQubitGate = self.num_qubits//2
        phi = phi_cte + randomness * np.pi * np.random.rand(6, num_TwoQubitGate)
        for m in range(num_TwoQubitGate):
            self.psi = two_qubit_gate_on_circuit(self.psi, twoqubit_U(phi[:, m]), 2 * m)       # This function applies two-qubit gate on circuit
        


    def odd_time_unitary(self, phi_cte = np.zeros((6,1)), randomness = 1., PBC = True) -> np.ndarray:
        """ Applies two-qubit U(1) charge-conserving unitary gates successively to neighboring qubits in odd time steps.

        Args:
            phi_cte: A constant (6-component) set of unitary phases. Default all 0.
            randomness: A float indicating the amount of randomness in unitary phases. Default is 1.
            PBC: A boolean indicating if we have periodic boundary condiction (PBC) or not.

        Returns:
            psi: The updated manybody state of N qubits. A 2^N-dimensional numpy array.

        """
        
        if PBC:        
            self.psi = self.psi.reshape(2 , 2**(self.num_qubits-1)).transpose(1, 0)
            self.even_time_unitary(phi_cte, randomness)
            self.psi = self.psi.reshape(2**(self.num_qubits-1), 2).transpose(1, 0)
        
        else:
            num_TwoQubitGate = self.num_qubits//2 - 1    # The number of gates for OBC case.
            phi = phi_cte + randomness * np.pi * np.random.rand(6, num_TwoQubitGate)

            for m in range(num_TwoQubitGate):
                self.psi = two_qubit_gate_on_circuit(self.psi, twoqubit_U(phi[:, m]), 2 * m + 1)
