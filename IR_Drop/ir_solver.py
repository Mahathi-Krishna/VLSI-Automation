# Main IR Solver:

import argparse

from file_transactions import *
from Netlist_Parser import *
from plots import *

# parser = argparse.ArgumentParser() 
# parser.add_argument("-input_file", help = "Argument for parsing the input", action = "store_true") # Argument for parsing the input
# parser.add_argument("-output_file", help = "Argument for the output", action = "store_true") # Argument for the output
# parser.add_argument('spice_netlist_name', help = 'Path to the input Spice netlist') # Path to the input Spice netlist
# parser.add_argument('voltage_file_name', help = 'Path to the output voltage file') # Path to the output voltage file

# args = parser.parse_args()
filename = './benchmarks/testcase1.sp' #args.spice_netlist_name
# outfilename = args.voltage_file_name

if 1==1: #args.voltage_file_name and args.spice_netlist_name:
    filedata = read_file(filename)
    parser = Netlist_Parser(filedata)
    # parser.solve_voltage_vector()
    # sp_g_matrix.write_voltage_vector(outfilename)
    form_current_map(parser.current_coord, parser.x_max, parser.y_max)
    # print(sorted(parser.metal_layers))
    form_ir_drop_map(parser.nodes, parser.v_vector, parser.x_max, parser.y_max)