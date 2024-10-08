'''
Convert .dat to .csv file
Please obtain the .dat file from the provided reference

Example:
python dat_to_cvs.py --input Xe.dat --output migdal_transition_Xe.csv
'''

import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Converting .dat file to .csv file")
parser.add_argument('--input', required=True, type=str, help="input file")
parser.add_argument('--output', required=True, type=str, help="name output file")
args = parser.parse_args()

if not os.path.exists(args.input):
    raise ValueError('file %s does not exist!' % (args.input))
if os.path.exists(args.output):
    raise ValueError('file %s already exists!' % (args.output))
assert args.output[-4:] == '.csv', "output file must be .csv"

file = open(args.input, 'r')
headers = []
result = {}
header_i = -1
result['E'] = []

for i, line in enumerate(file):
    if "Principal QN n" in line:
        header_i = i
    if i - header_i == 1:
        header = line.replace(" ", "")
        header = header[0] + '_' + header.strip('\n')[-1]
        headers.append(header)
        result[header] = []
    if i >= header_i + 3:
        result[header].append(line.split(' ')[-1].strip('\n'))
        if len(headers) == 1:
            result['E'].append(line.split(' ')[2])

df = pd.DataFrame()

for header in headers + ['E']:
    df[header] = result[header]

df.to_csv(args.output, index=False)
print('Done writing %s' % args.output)
