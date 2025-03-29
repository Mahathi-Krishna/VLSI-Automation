import re
import pandas as pd

path = "C:\\Users\\DELL\\Downloads\\ASU\\EEE_598 - VAuto\\Project\\Mini-proj 1\\"
filename = "c7552.bench"
file = path+filename

circuitfile = 'circuit_details.txt'
data = []                           # holds the file data
circuit_input_lines = {}            # holds the input line and type details
circuit_intermediate_outputs = {}   # holds the intermediate line details
circuit_output_lines = {}           # holds the output line details
gate_count_dict = {}                # holds the count for each gate type
gate_obj_list = []                  # stores the node names
gate_id_list = []                   # Stores the gate node numbers
gate_type_list = []                 # stores the type of each gate
gate_input_list = []                # stores the inputs of each gate as list of list

# Class for gates:
class Node:
    def __init__(self,name):
        self.gate = ""          # gate type
        self.name = name        # gate name
        self.outname = ""       # output node name
        self.cload = 0.0        # load capacitance
        self.inputs = []        # fanin details
        self.outputs = []       # fanout details
        self.tau_in = []
        self.tau_out = 0.0
        self.in_arr_time = []
        self.out_arr_time = []
        self.max_out_arr_time = 0.0

# Function to write to a file:
def fn_w_circuit_file(is_truncate, file_name, mode, data):
    with open(file_name, mode) as file:
        if is_truncate:
            file.truncate(0)
        file.write(data)

# Get input and output details:
def fn_io_parser(lines):
    str_data = ""
    for line in lines:
        if line.strip() and not line.startswith('#'):
            line_detail = [s.strip() for s in re.split(r"(\()", line)]

            if (line_detail[0] == "INPUT"):
                circuit_input_lines[line_detail[2].split(")")[0]] = line_detail[0].strip()
            
            if (line_detail[0] == "OUTPUT"):
                circuit_output_lines[line_detail[2].split(")")[0]] = line_detail[0].strip()
            
            if (line_detail[0] not in "INPUT|OUTPUT"):
                line_detail = line_detail[0].split("=")
                circuit_intermediate_outputs[line_detail[0].strip()] = line_detail[1].strip()

            str_data = (f"{len(circuit_input_lines)} primary inputs\n"
                        f"{len(circuit_output_lines)} primary outputs\n")
    
    fn_w_circuit_file(1, circuitfile, 'a', str_data)

# Get gate count and create nodes:
def fn_gate_detail_parser(lines):
    str_data = ""
    
    for line in lines:
        if line.strip() and not line.startswith('#'):
            line_detail = [s.strip() for s in re.split(r"(=)", line)]

            if (line_detail[0].split("(")[0] not in ("INPUT|OUTPUT")):
                # str_data = str_data + str(line_detail) + '\n'
                gate_out = line_detail[0]
                gate_type = line_detail[2].split("(")[0]
                gate_name = line_detail[2].split("(")[0] + '-' + line_detail[0]

                # Count number of gates by type
                if gate_type in gate_count_dict.keys():
                    gate_count_dict[gate_type] += 1
                else:
                    gate_count_dict[gate_type] = 1
                
                gate_id_list.append(gate_out)
                gate_type_list.append(line_detail[2].split("(")[0])
                gate_input_list.append([s.strip() for s in line_detail[2].split("(")[1].split(")")[0].split(",")])

    # Create nodes
    for i in range(len(gate_id_list)):
        gate_name = gate_type_list[i] + '-' + gate_id_list[i]
        gate = Node(gate_name)
        gate_obj_list.append(gate)
        gate.gate = gate_type_list[i]
        gate.outname = gate_id_list[i]
        gate.inputs = gate_input_list[i]

    for key in gate_count_dict.keys():
        str_data = str_data + f"{gate_count_dict[key]} {key} gates\n"

    fn_w_circuit_file(0, circuitfile, 'a', str_data)

# Function for fanin:
def fn_fanin_parser():
    str_data = "\nFanin...\n"
    for index, gate in enumerate(gate_obj_list):
        str_data = str_data + gate.name + ':'
        for i in gate.inputs:
            if (i in circuit_input_lines.keys()):
                str_data = str_data + ' ' + circuit_input_lines[i] + '-' + i + ','
            elif (i in circuit_intermediate_outputs.keys()):
                str_data = str_data + ' ' + circuit_intermediate_outputs[i] + '-' + i + ','
        str_data = str_data.strip(',') + '\n'
        
    fn_w_circuit_file(0, circuitfile, 'a', str_data)

# Function for fanout:
def fn_fanout_parser():
    intermediate_key_list = list(circuit_intermediate_outputs.keys())
    output_key_list = list(circuit_output_lines.keys())
    str_data = "\nFanout...\n"
    
    for gate_id in gate_id_list:
        str_data = str_data + circuit_intermediate_outputs[gate_id] + '-' + gate_id + ':'
        if gate_id not in output_key_list:
            for index, input in enumerate(gate_input_list):
                if gate_id in input:
                    id = intermediate_key_list[index]
                    gate_type = circuit_intermediate_outputs[id]
                    str_data = str_data + ' ' + gate_type + '-' + id + ','
        
        if (gate_id in output_key_list):
           gate_type = circuit_output_lines[gate_id]
           str_data = str_data + ' ' + gate_type + '-' + gate_id + ','
        str_data = str_data.strip(',') + '\n'
    
    fn_w_circuit_file(0, circuitfile, 'a', str_data)

with open(file, "r") as file:
    data = file.readlines()

fn_io_parser(data)
fn_gate_detail_parser(data)
fn_fanin_parser()
fn_fanout_parser()