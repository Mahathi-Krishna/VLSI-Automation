# Main IR Solver:

import argparse
import numpy as np

from file_transactions import *
from Netlist_Parser import *
from Plots import *

# parser = argparse.ArgumentParser() 
# parser.add_argument("-input_file", help = "Argument for parsing the input", action = "store_true") # Argument for parsing the input
# parser.add_argument("-output_file", help = "Argument for the output", action = "store_true") # Argument for the output
# parser.add_argument('spice_netlist_name', help = 'Path to the input Spice netlist') # Path to the input Spice netlist
# parser.add_argument('voltage_file_name', help = 'Path to the output voltage file') # Path to the output voltage file

# args = parser.parse_args()
filename = 'testcase1.sp' #args.spice_netlist_name
filepath = f"./Benchmarks/{filename}"
# outfilename = args.voltage_file_name

if 1==1: #args.voltage_file_name and args.spice_netlist_name:
    filedata = read_file(filepath)
    parser = Netlist_Parser(filedata)
    
    filename = filename.split('.')[0]

    parser.Process_Current_Map()
    # Custom_Plot(map, "Current Map")

    parser.Process_IR_Drop()
    # Custom_Plot(parser.ir_drop_mat, "IR Drop")

    parser.Process_Volt_Dist()
    # Custom_Plot(parser.eff_dist_volt, "Effective Distance to Voltage Source Map")
    
    parser.Process_PDN_Map()
    # Custom_Plot_PDN(parser.pdn_density_map, "PDN Density Map")

    # Save the matrices for model training:
    np.savez_compressed(
        f'{filename}.npz',
        current_map = parser.current_map,
        effective_voltage_dist_map = parser.eff_dist_volt,
        pdn_map = parser.pdn_density_map,
        ir_drop_map = parser.ir_drop_mat
    )