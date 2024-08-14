import subprocess
import copy
import os
from pathlib import Path
import pickle
import argparse
import yaml
import time


def create_sweep_files(dir, run_name, sweep_dict, settings):
    if len(sweep_dict) == 0:
        Path(os.path.join(dir, run_name)).mkdir(exist_ok=True)
        filename = os.path.join(dir, run_name, 'settings.yaml')
        if not os.path.exists(filename):
            with open(filename, "w") as stream:
                yaml.safe_dump(settings, stream)
        return [run_name]
    key = next(iter(sweep_dict.keys()))
    run_names = []
    for val in sweep_dict[key]:
        settings_copy = copy.deepcopy(settings)
        for m in settings_copy.keys():
            if key in settings_copy[m].keys():
                settings_copy[m][key] = val
        run_name_copy = run_name + '_' + key + str(val)
        sweep_dict_copy = copy.deepcopy(sweep_dict)
        sweep_dict_copy.pop(key)
        run_names_to_add = create_sweep_files(dir, run_name_copy, sweep_dict_copy, settings_copy)
        run_names = run_names + run_names_to_add
    return run_names



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep_dir", required=True, help="Directory where sweep settings lie"
    )
    parser.add_argument(
        "--run_name", required=True, help="Name of the sweep and directory of main settings file"
    )

    parser.add_argument(
        "--run_dir", required=True
    )

    parser.add_argument("--start_runs", action='store_true')

    parser.add_argument("--create_files", action='store_true')


    args = parser.parse_args()

    sweep_dict ={'d': [5], 'n': [50], 'k': [2]}

    with open(os.path.join(args.sweep_dir, args.run_name, 'settings.yaml'), "r") as stream:
        settings = yaml.safe_load(stream)

    with open(os.path.join(args.sweep_dir, args.run_name, 'sweep.yaml'), "r") as stream:
        sweep_dict = yaml.safe_load(stream)
    settings['data']['name'] = args.run_name
    if args.create_files:
        run_name_list = create_sweep_files(args.run_dir, args.run_name, sweep_dict, settings)
        with open(os.path.join(args.sweep_dir, args.run_name, 'run_names.txt'), 'w') as f:
            for line in run_name_list:
                f.write(f"{line}\n")

    if args.start_runs:
        runs = settings['data']['runs']
        with open(os.path.join(args.sweep_dir, args.run_name, 'run_names.txt'), 'r') as f:
            run_name_list = f.read().splitlines()
        for r in run_name_list:
            arg = r + ' ' + str(runs)
            # start shell script that calls the cluster submission file with arguments r and runs, needs to be adapted
            call_script_command = "./start_sweep.sh {}".format(arg)
            subprocess.check_call(call_script_command, shell=True)