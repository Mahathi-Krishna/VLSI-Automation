# Main IR Solver:

import argparse

from file_transactions import *
from Netlist_Parser import *
from Plots import *

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
    parser.Process_Current_Map()
    parser.Process_IR_Drop()
    parser.Process_Volt_Dist()
    Custom_Plot(parser.current_map, "Current Map")
    Custom_Plot(parser.ir_drop_mat, "IR Drop")
    Custom_Plot(parser.eff_dist_volt, "Effective Distance to Voltage Source Map")
    # print(sorted(parser.metal_layers))