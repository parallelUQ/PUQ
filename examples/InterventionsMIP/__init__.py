from pathlib import Path
import json

# Paths
project_path = Path(__file__).parent
instances_path = project_path / 'instances/'
reporting_path = project_path / 'reporting/'
output_path = project_path / 'output/'
plots_path = project_path / 'plots/'
config_path = project_path / 'config/'


def change_paths(args):
    if args.machine == 'frontera':  # TACC server
        global output_path
        global plots_path
        output_path = Path('$SCRATCH/InterventionsMIP/output') / 'output/'
        plots_path = Path('$SCRATCH/InterventionsMIP/output') / 'plots/'


# Read config files by default
# config = {}
# config_file = config_path / 'trigger_config_austin.json'
# with open(config_file, 'r') as input_file:
#     config = json.load(input_file)


def load_config_file(config_filename):
    global config
    config_file = config_path / config_filename
    with open(config_file, 'r') as input_file:
        config = json.load(input_file)


import logging

logger = logging.getLogger('TriggerOpt')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
