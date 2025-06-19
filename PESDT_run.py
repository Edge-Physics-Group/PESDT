from core import AnalyseSynthDiag
import json, os, sys
import argparse
import logging


def run_PESDT(input_dict_str):

    with open(input_dict_str, mode='r', encoding='utf-8') as f:
        # Remove comments
        with open("temp.json", 'w') as wf:
            for line in f.readlines():
                if line[0:2] == '//' or line[0:1] == '#':
                    continue
                wf.write(line)

    with open("temp.json", 'r') as f:
        input_dict = json.load(f)

    os.remove('temp.json')

    AnalyseSynthDiag(input_dict)

if __name__=='__main__':
    # Create the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the global logging level

    # Remove any existing handlers (important!)
    logger.handlers = []

    # File handler
    file_handler = logging.FileHandler('PESDT.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Console (stream) handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    # Add both handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.basicConfig(filename='PESDT.log', level=logging.INFO)
    logger.info('PESDT started')
    

    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Run PESDT')
    parser.add_argument('input_dict')
    args = parser.parse_args()

    # Handle the input arguments
    input_dict_file = args.input_dict

    if os.path.isfile(input_dict_file):
        logger.info(f"Found input dictionary: {input_dict_file}")
        run_PESDT(input_dict_file)
    else:
        logger.info(input_dict_file + ' not found')
        sys.exit(input_dict_file + ' not found')

    logger.info('Finished')
