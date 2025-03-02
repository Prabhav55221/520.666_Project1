# Project 1: EN.520.666-S25

> **Submission Details**
>
> Prabhav Singh (psingh54@jhu.edu)

### Project Structure

```os
- src

    - outputs (Plots for Both HMMs Init)
        - ORIGINAL_HMM_ARCHITECTURE (2, 4 States)
        - ALTERNATE_HMM_ARCHITECTURE (2, 4 States)

    - hmm.py    # Main Class for HMM
    - utils.py  # Helpers for data and initialization
    - main.py   # Argparse for CMD line calling

- environment.yml   # Create Python Env
- discussion.pdf    # LaTeX for discussions and answes
- README.pdf        # This file
- proj1.pdf         # Questions
```

### Installation

I have added the `environment.yml` file for help. Setting up the environment is as simple as running the below command:

```bash
conda env create -f environment.yml
conda activate project1_hmm
```

### Running the Experiments

You can run the experiments by following the commands below. Note that you can use argparse to change what data/hmm/states you use. Argparse is defined below:

```bash
usage: main.py [-h] [--num_states {2,4}] [--max_iterations MAX_ITERATIONS] [--hmm_type {standard,alternate}] [--train_file TRAIN_FILE] [--test_file TEST_FILE]

Run HMM experiments with different configurations.

options:
  -h, --help            show this help message and exit
  --num_states {2,4}    Number of hidden states in the HMM (2 or 4).
  --max_iterations MAX_ITERATIONS
                        Maximum iterations for the Baum-Welch algorithm.
  --hmm_type {standard,alternate}
                        Type of initialization for the HMM (standard or alternate).
  --train_file TRAIN_FILE
                        Path to the training data file.
  --test_file TEST_FILE
                        Path to the test data file.
```

For the purposed of this project, the 4 experiments can be run with the following commands:

```bash
cd src

# STANDARD INIT
python main.py --num_states 2 --max_iterations 600 --hmm_type standard
python main.py --num_states 4 --max_iterations 600 --hmm_type standard

# ALTERNATE INIT
python main.py --num_states 2 --max_iterations 600 --hmm_type alternate
```

Please read the attached PDF file for analysis.