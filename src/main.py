import argparse
import numpy as np
import matplotlib.pyplot as plt
import string
import json
import rich
from typing import Tuple, Dict, List, Optional

# Internal Imports
from utils import *
from hmm import HMM

# Constants
EPSILON = 1e-6

def main():
    """Main function to run the HMM experiments with argparse."""
    parser = argparse.ArgumentParser(description="Run HMM experiments with different configurations.")
    parser.add_argument("--num_states", type=int, choices=[2, 4], default=2,
                        help="Number of hidden states in the HMM (2 or 4).")
    parser.add_argument("--max_iterations", type=int, default=600,
                        help="Maximum iterations for the Baum-Welch algorithm.")
    parser.add_argument("--hmm_type", type=str, choices=["standard", "alternate"], default="standard",
                        help="Type of initialization for the HMM (standard or alternate).")
    parser.add_argument("--train_file", type=str, default="textA.txt",
                        help="Path to the training data file.")
    parser.add_argument("--test_file", type=str, default="textB.txt",
                        help="Path to the test data file.")
    
    args = parser.parse_args()

    print(f"Starting HMM experiment with {args.num_states} states and {args.max_iterations} iterations")
    
    # Load data
    try:
        train_data = load_data(args.train_file)
        test_data = load_data(args.test_file)
        print(f"Loaded training data: {len(train_data)} symbols")
        print(f"Loaded test data: {len(test_data)} symbols")
    except FileNotFoundError:
        print(f"Error: Could not find data files {args.train_file} or {args.test_file}")
        return
    
    # Initialize HMM based on type
    if args.hmm_type == "standard":
        print("\nRunning experiment with standard initialization...")
        if args.num_states == 2:
            transition_prob, emission_prob = initialize_hmm_2state()
        else:
            transition_prob, emission_prob = initialize_hmm_4state()
        foldername = './outputs/ORIGINAL_HMM_ARCHITECTURE/'
    else:
        print("\nRunning experiment with alternate initialization...")
        transition_prob, emission_prob = initialize_hmm_alternate(train_data)
        foldername = './outputs/ALTERNATE_HMM_ARCHITECTURE/'
    
    # Create and initialize HMM
    hmm = HMM(args.num_states, 27)
    hmm.initialize(transition_prob, emission_prob)
    
    # Run Baum-Welch algorithm
    print(f"Running Baum-Welch algorithm with {args.hmm_type} initialization...")
    info_dict = hmm.baum_welch(train_data, test_data, args.max_iterations)
    
    # Visualize results
    print("Generating visualizations...")
    output_prefix = foldername + f"hmm_{args.num_states}state_{args.hmm_type}_"
    hmm.visualize_results(info_dict, output_prefix)
    
    # Plot log-likelihood
    plt.figure(figsize=(10, 6))
    plt.plot(info_dict['iteration'], info_dict['train_log_likelihood'], label='Train Log-Likelihood')
    plt.plot(info_dict['iteration'], info_dict['test_log_likelihood'], label='Test Log-Likelihood')
    plt.xlabel('Iteration')
    plt.ylabel('Average Log-Likelihood')
    plt.title(f'HMM Training Log-Likelihood ({args.hmm_type.capitalize()} Initialization)')
    plt.legend()
    plt.grid(True)

    if args.hmm_type == 'standard':
        plt.savefig(f"./outputs/ORIGINAL_HMM_ARCHITECTURE/hmm_{args.num_states}state_{args.hmm_type}_log_likelihood.png")
        plt.close()
        if args.num_states == 2:
            with open('./outputs/STORED_LL/two_state_standard.json', 'w') as file:
                json.dump(info_dict, file, indent=4)
    else:
        plt.savefig(f"./outputs/ALTERNATE_HMM_ARCHITECTURE/hmm_{args.num_states}state_{args.hmm_type}_log_likelihood.png")
        plt.close()

        with open('./outputs/STORED_LL/two_state_standard.json', 'r') as file:
            info_dict_standard = json.load(file)

        # Compare results
        print("\nComparison of final log-likelihoods:")
        print(f"Standard initialization - Train: {info_dict_standard['train_log_likelihood'][-1]:.6f}, "
            f"Test: {info_dict_standard['test_log_likelihood'][-1]:.6f}")
        print(f"Alternate initialization - Train: {info_dict['train_log_likelihood'][-1]:.6f}, "
            f"Test: {info_dict['test_log_likelihood'][-1]:.6f}")
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.plot(info_dict_standard['iteration'], info_dict_standard['train_log_likelihood'], 
                label='Standard Init - Train')
        plt.plot(info_dict_standard['iteration'], info_dict_standard['test_log_likelihood'], 
                label='Standard Init - Test')
        plt.plot(info_dict['iteration'], info_dict['train_log_likelihood'], 
                label='Alternate Init - Train')
        plt.plot(info_dict['iteration'], info_dict['test_log_likelihood'], 
                label='Alternate Init - Test')
        plt.xlabel('Iteration')
        plt.ylabel('Average Log-Likelihood')
        plt.title('Comparison of Initialization Methods')
        plt.legend()
        plt.grid(True)
        plt.savefig("./outputs/hmm_initialization_comparison.png")
        plt.close()
        
    print("HMM experiment completed successfully!")

if __name__ == "__main__":
    main()