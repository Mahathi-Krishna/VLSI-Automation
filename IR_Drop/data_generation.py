# Main Data Generation:
import argparse
import os

from file_transactions import *
from netlist_parser import *
from plotter import *
from data_augment import *

# Function to parse the spice netlist and generate the csv and voltage files:
def Data_Generator(input_path, output_path, feature_path, label_path, mode, gen_voltage_file=False):
    filename = input_path
    output_path = output_path
    os.makedirs(output_path, exist_ok=True)

    filedata = Read_File(filename)
    parser = Netlist_Parser(filedata)

    filename = os.path.basename(filename)
    filename = os.path.splitext(filename)[0].strip()
    input_dir = os.path.dirname(input_path)

    # Generate the Voltage file:
    # Voltage file will be generated during inference:
    if gen_voltage_file:
        parser.Write_Voltage_Vector(os.path.join(input_dir, filename))

    # Process Current, Voltage Source, PDN and IR Drop distributions:
    parser.Process_Current_Map()
    parser.Process_IR_Drop(os.path.join(input_dir, f"{filename}.voltage"))
    parser.Process_Volt_Dist()
    parser.Process_PDN_Map()

    # Save Current, Voltage Source, PDN and IR Drop distributions as csv and npy files:
    Data_Augment(filename, output_path, feature_path, label_path,
                    parser.current_map, parser.eff_dist_volt, parser.pdn_density_map, parser.ir_drop_mat, train_or_test=mode)


# Main function call - from CLI:
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("-spice_netlist", help = "Input: Path to Spice netlist", type = str, required = True)
    parser.add_argument("-voltage_file", help = "Input: Path to Voltage txt file", type = str, required = True)
    parser.add_argument("-output", help = "Output: Path where four csv files are stored", type = str, required = True)

    args = parser.parse_args()

    if args.spice_netlist and args.voltage_file and args.output:
        filename = args.spice_netlist       # Spice netlist path/name
        voltage_file = args.voltage_file    # Voltage file path/name
        output_path = args.output           # Path where the four CSV files will be saved

        os.makedirs(output_path, exist_ok=True)

        test_feature_path = os.path.join(output_path, "Features")
        test_label_path = os.path.join(output_path, "Labels")
        
        os.makedirs(test_feature_path, exist_ok=True)
        os.makedirs(test_label_path, exist_ok=True)

        filedata = Read_File(filename)
        parser = Netlist_Parser(filedata)

        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0].strip()
        print(filename)

        # Process Current, Voltage Source, PDN and IR Drop distributions:
        parser.Process_Current_Map()
        parser.Process_IR_Drop(voltage_file)
        parser.Process_Volt_Dist()
        parser.Process_PDN_Map()

        # Save Current, Voltage Source, PDN and IR Drop distributions as csv and npy files:
        Data_Augment(filename, output_path, test_feature_path, test_label_path,
                     parser.current_map, parser.eff_dist_volt, parser.pdn_density_map, parser.ir_drop_mat, train_or_test='test')