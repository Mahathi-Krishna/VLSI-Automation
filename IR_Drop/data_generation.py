# Main IR Solver:

# /Features > rotated, .npy & same matrix size
# /Labels > rotated, .npy & same matrix size
# Predictnpz > .npz

import argparse
import numpy as np
import sys
import cv2
import os

from file_transactions import *
from Netlist_Parser import *
from Plotter import *

# parser = argparse.ArgumentParser() 
# parser.add_argument("-input_file", help = "Argument for parsing the input", action = "store_true") # Argument for parsing the input
# parser.add_argument("-output_file", help = "Argument for the output", action = "store_true") # Argument for the output
# parser.add_argument('spice_netlist_name', help = 'Path to the input Spice netlist') # Path to the input Spice netlist
# parser.add_argument('voltage_file_name', help = 'Path to the output voltage file') # Path to the output voltage file

# args = parser.parse_args()
filename = sys.argv[1] #args.spice_netlist_name
feature_path = "./Without_Filter/Test_Features"
label_path = "./Without_Filter/Test_Labels"

# filepath = f"./Benchmarks/{filename}"
# outfilename = args.voltage_file_name

if 1==1: #args.voltage_file_name and args.spice_netlist_name:
    filedata = read_file(filename)
    parser = Netlist_Parser(filedata)
    
    filename = filename.split('.')[1].split('\\')[1]
    print(filename)

    parser.Process_Current_Map()
    current_map = cv2.resize(parser.current_map, (256, 256), interpolation=cv2.INTER_AREA)
    current_map_90 = np.rot90(current_map, k=-1)
    current_map_m90 = np.rot90(current_map, k=1)
    current_map_180 = np.rot90(current_map, k=2)
    # Custom_Plot(current_map, "Current Map")

    parser.Process_IR_Drop()
    ir_drop_map = cv2.resize(parser.ir_drop_mat, (256, 256), interpolation=cv2.INTER_AREA)
    ir_drop_map_90 = np.rot90(ir_drop_map, k=-1)
    ir_drop_map_m90 = np.rot90(ir_drop_map, k=1)
    ir_drop_map_180 = np.rot90(ir_drop_map, k=2)
    # Custom_Plot(ir_drop_map, "IR Drop")

    parser.Process_Volt_Dist()
    eff_dist_volt_map = cv2.resize(parser.eff_dist_volt, (256, 256), interpolation=cv2.INTER_AREA)
    eff_dist_volt_map_90 = np.rot90(eff_dist_volt_map, k=-1)
    eff_dist_volt_map_m90 = np.rot90(eff_dist_volt_map, k=1)
    eff_dist_volt_map_180 = np.rot90(eff_dist_volt_map, k=2)
    # Custom_Plot(eff_dist_volt_map, "Effective Distance to Voltage Source Map")
    
    parser.Process_PDN_Map()
    pdn_density_map = cv2.resize(parser.pdn_density_map, (256, 256), interpolation=cv2.INTER_AREA)
    pdn_density_map_90 = np.rot90(pdn_density_map, k=-1)
    pdn_density_map_m90 = np.rot90(pdn_density_map, k=1)
    pdn_density_map_180 = np.rot90(pdn_density_map, k=2)
    # Custom_Plot_PDN(pdn_density_map, "PDN Density Map")


    # Save the normal matrices for model training:
    output_path = os.path.join(feature_path, f"{filename}.npy")
    np.save(output_path,
        np.stack([
            current_map,                    # index 0 - current map
            eff_dist_volt_map,              # index 1 - eff voltage distance map
            pdn_density_map,                # index 2 - pdn map
            ], axis=0).astype(np.float32))

    output_path = os.path.join(label_path, f"{filename}.npy")
    np.save(output_path, ir_drop_map.astype(np.float32))


    # Save the flipped versions of normal matrices:
    output_path = os.path.join(feature_path, f"{filename}_ud.npy")
    np.save(output_path,
        np.stack([
            np.flipud(current_map),
            np.flipud(eff_dist_volt_map),
            np.flipud(pdn_density_map),
            ], axis=0).astype(np.float32))
    
    output_path = os.path.join(label_path, f"{filename}_ud.npy")
    np.save(output_path, np.flipud(ir_drop_map).astype(np.float32))


    output_path = os.path.join(feature_path, f"{filename}_lr.npy")
    np.save(output_path,
        np.stack([
            np.fliplr(current_map),
            np.fliplr(eff_dist_volt_map),
            np.fliplr(pdn_density_map),
            ], axis=0).astype(np.float32))
    
    output_path = os.path.join(label_path, f"{filename}_lr.npy")
    np.save(output_path, np.fliplr(ir_drop_map).astype(np.float32))


    # Save the rotated versions of normal matrices:
    output_path = os.path.join(feature_path, f"{filename}_1.npy")
    np.save(output_path,
        np.stack([
            current_map_90,
            eff_dist_volt_map_90,
            pdn_density_map_90,
            ], axis=0).astype(np.float32))

    output_path = os.path.join(label_path, f"{filename}_1.npy")
    np.save(output_path, ir_drop_map_90.astype(np.float32))


    output_path = os.path.join(feature_path, f"{filename}_2.npy")
    np.save(output_path,
        np.stack([
            current_map_m90,
            eff_dist_volt_map_m90,
            pdn_density_map_m90,
            ], axis=0).astype(np.float32))

    output_path = os.path.join(label_path, f"{filename}_2.npy")
    np.save(output_path, ir_drop_map_m90.astype(np.float32))


    output_path = os.path.join(feature_path, f"{filename}_3.npy")
    np.save(output_path,
        np.stack([
            current_map_180,
            eff_dist_volt_map_180,
            pdn_density_map_180,
            ], axis=0).astype(np.float32))

    output_path = os.path.join(label_path, f"{filename}_3.npy")
    np.save(output_path, ir_drop_map_180.astype(np.float32))