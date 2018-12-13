# Master script to run all individual sickle-cell-related scripts (in order)

import os
import re
import importlib
import utils_sickle_stats as utils

# Capture command-line arguments (to be shared between all modules)
args = utils.parse_arguments()

# Initialize common logger
utils.initialize_logger(args)

# Grab list of all sickle cell modules of the form "A12*.py" and import them.
p = re.compile('^A\d\d.*\.py$')
sickle_dir = '.'
sickle_modules = [f[:-3] for f in os.listdir(
    sickle_dir) if os.path.isfile(os.path.join(sickle_dir, f)) and p.match(f)]
sickle_modules = sorted(sickle_modules)

# Import and run each module
for module_name in sickle_modules:
    print(module_name)
    module = importlib.import_module(module_name)
    getattr(module, 'main')(args)

#map(__import__, sickle_modules)

# Run "main(args)" in each sickle cell module, with modules run in order
#for module in sickle_modules:
#    import IPython
#    IPython.embed()
#    getattr(globals()[module], 'main')(args)
