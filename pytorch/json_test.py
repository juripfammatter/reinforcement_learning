import json
import sys
import os

if __name__ == "__main__":
    """ Import JSON file and check existence"""
    config_file = sys.argv[1]
    if os.path.exists(config_file):
        print(f"Using {config_file} as config file")

    with open(config_file, "r") as file:
        config = json.load(file)

    print(f"Config file loaded: {config_file} with config:\n{json.dumps(config, indent=4)}")
