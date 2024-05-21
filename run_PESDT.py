from PESDT.analyse import AnalyseSynthDiag
import json, os, sys
import argparse

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

    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Run PESDT')
    parser.add_argument('input_dict')
    args = parser.parse_args()

    # Handle the input arguments
    input_dict_file = args.input_dict

    if os.path.isfile(input_dict_file):
        print('Found input dictionary: ', input_dict_file)
        run_PESDT(input_dict_file)
    else:
        sys.exit(input_dict_file + ' not found')
