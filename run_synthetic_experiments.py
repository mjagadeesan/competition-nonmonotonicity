#!/usr/bin/env python3
import sys
import synthetic_helper

def run_functions(trial_num, num_players, solution, setting_no, dist_arg):
    # Run additional code based on conditions
    if solution == 'bayes' and setting_no == 1:
        synthetic_helper.compute_bayes_optimal_setting1(dist_arg, trial_num=trial_num)

    elif solution == 'bayes' and setting_no == 2:
        synthetic_helper.compute_bayes_optimal_setting2(dist_arg, trial_num=trial_num)

    elif solution == 'bayes' and setting_no == 3:
        synthetic_helper.compute_bayes_optimal_setting3(dist_arg, trial_num=trial_num)

    elif solution == 'nash' and setting_no == 1:
        synthetic_helper.compute_nash_equilibria_setting1(num_players, dist_arg, trial_num=trial_num)

    elif solution == 'nash' and setting_no == 2:
        synthetic_helper.compute_nash_equilibria_setting2(num_players, dist_arg, trial_num=trial_num)

    elif solution == 'nash' and setting_no == 3:
        synthetic_helper.compute_nash_equilibria_setting3(num_players, int(dist_arg), trial_num=trial_num)

    else:
        print("Invalid combination of solution and classes")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python script.py trial_num num_players solution setting_no dist_arg")
        sys.exit(1)

    trial_num = int(sys.argv[1])
    num_players = int(sys.argv[2])
    solution = sys.argv[3]
    setting_no = int(sys.argv[4])
    dist_arg = float(sys.argv[5])

    run_functions(trial_num, num_players, solution, setting_no, dist_arg)

