#!/usr/bin/env python3
import sys
import cifar_helper

def run_functions(trial_num, num_players, solution, classes):
    # Run additional code based on conditions
    if solution == 'bayes' and classes == 'multi_class':
        cifar_helper.compute_bayes_optimal_multiclass(trial_num)

    elif solution == 'bayes' and classes == 'binary':
        cifar_helper.compute_bayes_optimal_binary(trial_num)

    elif solution == 'nash' and classes == 'binary':
        cifar_helper.compute_nash_equilibria_binary(num_players, trial_num)

    elif solution == 'nash' and classes == 'multi_class':
        cifar_helper.compute_nash_equilibria_multiclass(num_players, trial_num)

    else:
        print("Invalid combination of solution and classes")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python script.py trial_num num_players solution classes")
        sys.exit(1)

    trial_num = int(sys.argv[1])
    num_players = int(sys.argv[2])
    solution = sys.argv[3]
    classes = sys.argv[4]

    run_functions(trial_num, num_players, solution, classes)

