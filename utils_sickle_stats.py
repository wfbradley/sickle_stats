import os
import sys
import logging
import argparse
from version_sickle_stats import __version__

# Create directory only if needed.  (Makes nested directories if needed,
#   e.g. "new1/new2/new3/".)
def safe_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


logger = logging.getLogger('Sickle Cell Stats logger')
logger_initialized = False


def check_permissions(args):
    if os.getuid() == 0:
        logger.info('Running as root.')
        return
    # If here, we are not root.
    if args.halt:
        logger.info('Not running as root.')
        raise Exception
    return


def initialize_logger(args):
    global logger_initialized

    if logger_initialized:
        return
    logger_initialized = True

    safe_mkdir(args.working_dir)
    log_file = os.path.join(args.working_dir, 'run.log')

    if not (args.keep_old_logs):
        try:
            os.remove(log_file)
        except Exception:
            pass
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("##############  Beginning logging  ##############")
    logger.info("Sickle Stats version %s" % __version__)
    logger.info("Command line: %s" % (' '.join(sys.argv)))
    logger.info("Runtime parameters:")
    for f in args.__dict__:
        if f.startswith('__'):
            continue
        logger.info("   %s  :  %s" % (f, args.__dict__[f]))


def parse_arguments():

    parser = argparse.ArgumentParser(description='Supervised ML classification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--confidential_dir', type=str, default='confidential_data',
                        help='Directory for reading and writing confidential data.')
    parser.add_argument('--working_dir', type=str, default='data',
                        help='Directory for all working files and output')
    parser.add_argument('--input_file', type=str, default='l_voc_acs_mh.xlsx',
                        help='Input Excel data file within confidential_data_dir')
    parser.add_argument('--clean_file', type=str, default='cleaned_data.csv',
                        help='Input Excel data file within confidential_data_dir')
    parser.add_argument('--draw_plots', action='store_true', default=False,
                        help='Render plots to screen')

    parser.add_argument('--keep_old_logs', action='store_true', default=False,
                        help='By default, old log file is deleted; this preserves it.')
    args = parser.parse_args()

    # Fully qualify "~"s from path names
    args.confidential_dir = os.path.expanduser(args.confidential_dir)
    args.working_dir = os.path.expanduser(args.working_dir)

    return(args)
