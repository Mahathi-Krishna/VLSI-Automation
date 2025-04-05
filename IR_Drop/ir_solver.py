# Main IR Solver:

import argparse

from file_transactions import *
from Sparse_G_matrix import *

parser = argparse.ArgumentParser() 
parser.add_argument("-input_file", help = "Argument for parsing the input", action = "store_true") # Argument for parsing the input
parser.add_argument("-output_file", help = "Argument for the output", action = "store_true") # Argument for the output
parser.add_argument('spice_netlist_name', help = 'Path to the input Spice netlist') # Path to the input Spice netlist
parser.add_argument('voltage_file_name', help = 'Path to the output voltage file') # Path to the output voltage file

args = parser.parse_args()
filename = args.spice_netlist_name
outfilename = args.voltage_file_name

if args.voltage_file_name and args.spice_netlist_name:
    filedata = read_file(filename)
    sp_g_matrix = Sparse_G_matrix(filedata)
    sp_g_matrix.get_voltage_vector()
    sp_g_matrix.write_voltage_vector(outfilename)