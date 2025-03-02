"""
Hidden Markov Model Implementation for English Text Modeling

Author: Prabhav Singh
Email: psingh54@jhu.edu

Note: Claude on Cursor was used to generate docstrings and comments!
"""

import numpy as np
import matplotlib.pyplot as plt
import string
import rich
from typing import Tuple, Dict, List, Optional

# Internal Imports
from utils import *

# Constants
EPSILON = 1e-6

class HMM:
    """
    Hidden Markov Model implementation with Baum-Welch algorithm for parameter estimation.
    """
    
    def __init__(self, num_states: int, num_symbols: int):
        """
        Initialize the HMM with given number of states and output symbols.
        
        Args:
            num_states: Number of hidden states in the model
            num_symbols: Number of possible output symbols
        """

        self.num_states = num_states
        self.num_symbols = num_symbols
        
        # Probability matrices
        self.transition_prob = None
        self.emission_prob = None
        self.initial_prob = None
    
    def initialize(self, transition_prob: np.ndarray, emission_prob: np.ndarray, 
                  initial_prob: Optional[np.ndarray] = None):
        """
        Initialize the HMM with given probabilities.
        
        Args:
            transition_prob: Transition probability matrix [num_states x num_states]
            emission_prob: Emission probability matrix [num_states x num_symbols]
            initial_prob: Initial state probability vector [num_states]
        """

        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        
        # If no initial probability is provided, use uniform distribution
        if initial_prob is None:
            self.initial_prob = np.ones(self.num_states) / self.num_states
        else:
            self.initial_prob = initial_prob

        self._verify_probabilities()
    
    def _verify_probabilities(self):
        """
        Verify that probability matrices are valid.
        """

        # Check transition probabilities
        for s in range(self.num_states):
            assert abs(np.sum(self.transition_prob[s, :]) - 1.0) < EPSILON, \
                f"Transition probabilities from state {s} don't sum to 1"
        
        # Check emission probabilities
        for s in range(self.num_states):
            assert abs(np.sum(self.emission_prob[s, :]) - 1.0) < EPSILON, \
                f"Emission probabilities from state {s} don't sum to 1"
        
        # Check initial probabilities
        assert abs(np.sum(self.initial_prob) - 1.0) < EPSILON, \
            "Initial probabilities don't sum to 1"
        
        print('Probabilities are verified!')
    
    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform forward pass to compute alpha values.
        
        Args:
            observations: Sequence of observations [seq_length]
            
        Returns:
            alpha: Forward variables [num_states x seq_length]
            scale: Scaling factors [seq_length]
        """

        T = len(observations)
        alpha = np.zeros((self.num_states, T))
        scale = np.zeros(T)
        
        alpha[:, 0] = self.initial_prob * self.emission_prob[:, observations[0]]
        
        # Scale alpha to avoid numerical underflow
        scale[0] = np.sum(alpha[:, 0])
        if scale[0] > 0:
            alpha[:, 0] /= scale[0]
        
        # Recursive calculation
        for t in range(1, T):
            for j in range(self.num_states):
                alpha[j, t] = np.sum(alpha[:, t-1] * self.transition_prob[:, j]) * \
                             self.emission_prob[j, observations[t]]
            
            scale[t] = np.sum(alpha[:, t])
            if scale[t] > 0:
                alpha[:, t] /= scale[t]
        
        return alpha, scale
    
    def backward(self, observations: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """
        Perform backward pass to compute beta values.
        
        Args:
            observations: Sequence of observations [seq_length]
            scale: Scaling factors from forward pass [seq_length]
            
        Returns:
            beta: Backward variables [num_states x seq_length]
        """

        T = len(observations)
        beta = np.zeros((self.num_states, T))
        
        beta[:, T-1] = 1.0
        
        for t in range(T-2, -1, -1):
            for i in range(self.num_states):
                for j in range(self.num_states):
                    beta[i, t] += self.transition_prob[i, j] * \
                                 self.emission_prob[j, observations[t+1]] * \
                                 beta[j, t+1]
            
            if scale[t+1] > 0:
                beta[:, t] /= scale[t+1]
        
        return beta
    
    def baum_welch(self, train_data: np.ndarray, test_data: np.ndarray, 
                  max_iterations: int) -> Dict:
        """
        Estimate HMM parameters using the Baum-Welch algorithm.
        
        Args:
            train_data: Training data sequence
            test_data: Test data sequence
            max_iterations: Maximum number of iterations
            
        Returns:
            info_dict: Dictionary with learning progress information
        """

        # Initialize information dictionary to track learning progress
        info_dict = {
            'iteration': [],
            'train_log_likelihood': [],
            'test_log_likelihood': [],
        }
        
        for state in range(self.num_states):
            info_dict[f'emission_a_state_{state}'] = []
            info_dict[f'emission_n_state_{state}'] = []
        
        # Baum-Welch iterations
        for iteration in range(max_iterations):

            info_dict['iteration'].append(iteration)
            
            train_log_likelihood = self.log_likelihood(train_data) / len(train_data)
            test_log_likelihood = self.log_likelihood(test_data) / len(test_data)
            
            info_dict['train_log_likelihood'].append(train_log_likelihood)
            info_dict['test_log_likelihood'].append(test_log_likelihood)
            
            for state in range(self.num_states):
                info_dict[f'emission_a_state_{state}'].append(self.emission_prob[state, 0])
                info_dict[f'emission_n_state_{state}'].append(self.emission_prob[state, 13])
            
            print(f"Iteration {iteration}: Train LL = {train_log_likelihood:.6f}, "
                  f"Test LL = {test_log_likelihood:.6f}")
            
            alpha, scale = self.forward(train_data)
            beta = self.backward(train_data, scale)
            
            # Update HMM parameters
            self._update_parameters(train_data, alpha, beta)
        
        return info_dict
    
    def _update_parameters(self, observations: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
        """
        Update HMM parameters based on forward-backward calculations.
        
        Args:
            observations: Sequence of observations
            alpha: Forward variables
            beta: Backward variables
        """

        T = len(observations)
        
        gamma = alpha * beta
        
        for t in range(T):
            if np.sum(gamma[:, t]) > 0:
                gamma[:, t] = gamma[:, t] / np.sum(gamma[:, t])
        
        xi = np.zeros((T-1, self.num_states, self.num_states))
        
        for t in range(T-1):
            for i in range(self.num_states):
                for j in range(self.num_states):
                    xi[t, i, j] = alpha[i, t] * self.transition_prob[i, j] * \
                                 self.emission_prob[j, observations[t+1]] * beta[j, t+1]
            
            if np.sum(xi[t, :, :]) > 0:
                xi[t, :, :] = xi[t, :, :] / np.sum(xi[t, :, :])
        
        for i in range(self.num_states):
            gamma_sum_i = np.sum(gamma[i, :-1])
            
            for j in range(self.num_states):
                if gamma_sum_i > 0:
                    self.transition_prob[i, j] = np.sum(xi[:, i, j]) / gamma_sum_i
                else:
                    pass
        
        # Update emission probabilities
        for i in range(self.num_states):
            gamma_sum_i = np.sum(gamma[i, :])
            
            for k in range(self.num_symbols):
                gamma_sum_i_k = np.sum(gamma[i, observations == k])
                
                if gamma_sum_i > 0:
                    self.emission_prob[i, k] = gamma_sum_i_k / gamma_sum_i
                else:
                    pass
        
        # Re-normalize probabilities to ensure they sum to 1
        for i in range(self.num_states):
            if np.sum(self.transition_prob[i, :]) > 0:
                self.transition_prob[i, :] = self.transition_prob[i, :] / np.sum(self.transition_prob[i, :])
            
            if np.sum(self.emission_prob[i, :]) > 0:
                self.emission_prob[i, :] = self.emission_prob[i, :] / np.sum(self.emission_prob[i, :])
    
    def log_likelihood(self, observations: np.ndarray) -> float:
        """
        Compute the log-likelihood of an observation sequence.
        
        Args:
            observations: Sequence of observations
            
        Returns:
            log_likelihood: Log-likelihood of the sequence
        """
        _, scale = self.forward(observations)
        
        log_likelihood = np.sum(np.log(scale + EPSILON))
        return log_likelihood
    
    def visualize_results(self, info_dict: Dict, output_prefix: str = ""):
        """
        Visualize the results of Baum-Welch training.
        
        Args:
            info_dict: Dictionary with learning progress information
            output_prefix: Prefix for output filenames
        """
        
        # Plot log-likelihood curves
        plt.figure(figsize=(10, 6))
        plt.plot(info_dict['iteration'], info_dict['train_log_likelihood'], label='Training Data')
        plt.plot(info_dict['iteration'], info_dict['test_log_likelihood'], label='Test Data')
        plt.xlabel('Iteration')
        plt.ylabel('Average Log-Likelihood')
        plt.title('Log-Likelihood vs. Iteration')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_prefix}log_likelihood.png")
        plt.close()
        
        # Plot emission probabilities for letter 'a'
        plt.figure(figsize=(10, 6))
        for state in range(self.num_states):
            plt.plot(info_dict['iteration'], info_dict[f'emission_a_state_{state}'], 
                    label=f'State {state+1}')
        plt.xlabel('Iteration')
        plt.ylabel('Emission Probability for "a"')
        plt.title('Emission Probability for Letter "a" vs. Iteration')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_prefix}emission_a.png")
        plt.close()
        
        # Plot emission probabilities for letter 'n'
        plt.figure(figsize=(10, 6))
        for state in range(self.num_states):
            plt.plot(info_dict['iteration'], info_dict[f'emission_n_state_{state}'], 
                    label=f'State {state+1}')
        plt.xlabel('Iteration')
        plt.ylabel('Emission Probability for "n"')
        plt.title('Emission Probability for Letter "n" vs. Iteration')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_prefix}emission_n.png")
        plt.close()
        
        # Plot final emission probabilities for all letters in each state
        alphabet = list(string.ascii_lowercase) + ['#']
        
        for state in range(self.num_states):
            plt.figure(figsize=(12, 6))
            plt.bar(alphabet, self.emission_prob[state, :])
            plt.xlabel('Letters')
            plt.ylabel('Emission Probability')
            plt.title(f'Emission Probability Distribution for State {state+1}')
            plt.savefig(f"{output_prefix}emission_dist_state_{state+1}.png")
            plt.close()
            