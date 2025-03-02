"""
Helpers for Initialization and Data Loading

Author: Prabhav Singh
Email: psingh54@jhu.edu

Note: Claude on Cursor was used to generate docstrings and comments!
"""

import numpy as np
import matplotlib.pyplot as plt
import string
import rich
from typing import Tuple, Dict, List, Optional

# Constants
EPSILON = 1e-6

def load_data(filename: str) -> np.ndarray:
    """
    Load text data from file and convert to sequence of symbol indices.
    
    Args:
        filename: Path to the text file
        
    Returns:
        data: Array of symbol indices
    """

    alphabet = list(string.ascii_lowercase) + [' ']
    symbol_to_index = {symbol: idx for idx, symbol in enumerate(alphabet)}
    
    with open(filename, 'r') as file:
        content = file.read().strip()
    
    data = []
    for char in content:
        if char in symbol_to_index:
            data.append(symbol_to_index[char])
        else:
            pass
    
    return np.array(data)


def initialize_hmm_2state() -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize transition and emission probabilities for a 2-state HMM.
    
    Returns:
        transition_prob: Initial transition probability matrix
        emission_prob: Initial emission probability matrix
    """

    transition_prob = np.array([
        [0.49, 0.51],
        [0.51, 0.49]
    ])
    
    num_symbols = 27
    emission_prob = np.zeros((2, num_symbols))
    
    emission_prob[0, 0:13] = 0.0370
    emission_prob[0, 13:26] = 0.0371
    emission_prob[0, 26] = 0.0367
    
    emission_prob[1, 0:13] = 0.0371
    emission_prob[1, 13:26] = 0.0370 
    emission_prob[1, 26] = 0.0367 

    print("TRANSTIION PROBS INIT:")
    rich.print(transition_prob)

    print("\EMISSION PROBS INIT:")
    rich.print(emission_prob)
    
    return transition_prob, emission_prob


def initialize_hmm_4state() -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize transition and emission probabilities for a 4-state HMM.
    
    Returns:
        transition_prob: Initial transition probability matrix
        emission_prob: Initial emission probability matrix
    """

    # Initialize transition probabilities
    transition_prob = np.array([
        [0.24, 0.26, 0.24, 0.26],
        [0.26, 0.24, 0.26, 0.24],
        [0.26, 0.26, 0.24, 0.24],
        [0.24, 0.24, 0.26, 0.26]
    ])
    
    # Initialize emission probabilities
    num_symbols = 27 
    emission_prob = np.zeros((4, num_symbols))
    
    # Similar pattern to 2-state but with 4 states
    for i in range(4):
        if i % 2 == 0:
            emission_prob[i, 0:13] = 0.0370  
            emission_prob[i, 13:26] = 0.0371 
        else:
            emission_prob[i, 0:13] = 0.0371  
            emission_prob[i, 13:26] = 0.0370
        emission_prob[i, 26] = 0.0367 
    
    return transition_prob, emission_prob


def analyze_emission_probabilities(emission_prob: np.ndarray):
    """
    Analyze the learned emission probabilities to understand what the model learned.
    
    Args:
        emission_prob: Emission probability matrix after training
    """

    alphabet = list(string.ascii_lowercase) + ['#'] 
    
    print("\nAnalysis of learned emission probabilities:")
    
    # Find most probable letters for each state
    for state in range(emission_prob.shape[0]):
        top_indices = np.argsort(-emission_prob[state, :])[:5] 
        top_letters = [alphabet[idx] for idx in top_indices]
        top_probs = [emission_prob[state, idx] for idx in top_indices]
        
        print(f"State {state+1} most probable letters: " + 
              ", ".join([f"{letter}({prob:.4f})" for letter, prob in zip(top_letters, top_probs)]))
    
    # Find biggest differences between states (for 2-state model)
    if emission_prob.shape[0] == 2:
        differences = emission_prob[0, :] - emission_prob[1, :]
        top_diff_indices = np.argsort(-np.abs(differences))[:5] 
        
        print("\nBiggest differences between states:")
        for idx in top_diff_indices:
            print(f"Letter '{alphabet[idx]}': State 1 = {emission_prob[0, idx]:.4f}, " +
                  f"State 2 = {emission_prob[1, idx]:.4f}, " +
                  f"Difference = {differences[idx]:.4f}")
            
def initialize_hmm_alternate(train_data, random_seed=42):
    """
    Initialize HMM with alternate initialization based on letter frequencies and perturbation.
    
    Args:
        train_data: Training data sequence
        random_seed: Random seed for reproducibility
        
    Returns:
        transition_prob: Initial transition probability matrix
        emission_prob: Initial emission probability matrix
    """
    np.random.seed(random_seed)
    
    num_symbols = 27  
    letter_counts = np.zeros(num_symbols)
    for symbol in train_data:
        letter_counts[symbol] += 1
    
    letter_freq = letter_counts / np.sum(letter_counts)
    
    r_vector = np.random.rand(num_symbols)
    r_mean = np.mean(r_vector)
    delta = r_vector - r_mean 

    lambda_val = 0.001
    
    while not (np.all(letter_freq - lambda_val * delta > 0) and 
               np.all(letter_freq + lambda_val * delta > 0)):
        lambda_val *= 0.5
        
    print(f"Using lambda value: {lambda_val}")
    
    emission_prob = np.zeros((2, num_symbols))
    emission_prob[0, :] = letter_freq - lambda_val * delta
    emission_prob[1, :] = letter_freq + lambda_val * delta
    
    emission_prob[0, :] = emission_prob[0, :] / np.sum(emission_prob[0, :])
    emission_prob[1, :] = emission_prob[1, :] / np.sum(emission_prob[1, :])
    
    transition_prob = np.array([
        [0.49, 0.51],
        [0.51, 0.49] 
    ])
    
    return transition_prob, emission_prob