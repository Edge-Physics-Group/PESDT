from core import ProcessEdgeSim
import json, os, sys
import argparse
import logging


def run_PESDT(input_dict):
    ProcessEdgeSim(input_dict)

if __name__=='__main__':
    # Create the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the global logging level

    # Remove any existing handlers (important!)
    logger.handlers = []

    # File handler
    

    # Console (stream) handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))


    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Run PESDT')
    parser.add_argument('input_dict')
    args = parser.parse_args()

    # Handle the input arguments
    input_dict_file = args.input_dict

    if os.path.isfile(input_dict_file):
        with open(input_dict_file, mode='r', encoding='utf-8') as f:
            input_dict = json.load(f)
        dirs = os.path.join(input_dict["save_dir"], input_dict["job_name"])
        file_handler = logging.FileHandler( os.path.join( dirs,'PESDT.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.info('PESDT started')
        logger.info(f"Found input dictionary: {input_dict_file}")
        run_PESDT(input_dict)
    else:
        file_handler = logging.FileHandler('PESDT.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        logger.info('PESDT started')
        logger.info(input_dict_file + ' not found')
        sys.exit(input_dict_file + ' not found')

    logger.info('Finished')
