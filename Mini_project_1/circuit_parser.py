import re           #To parse throug the NLDM or the Circuit file 
import numpy as np  #To handle arrays
from nldm_parser import *

# phase-2
from collections import deque
# Parameters
circuitfile = 'ckt_details.txt'     #Output file for the bench parser
data = []                           # holds the file data
circuit_input_lines = {}            # holds the input line and type details
circuit_intermediate_outputs = {}   # holds the intermediate line details
circuit_output_lines = {}           # holds the output line details
gate_count_dict = {}                # holds the count for each gate type
gate_obj_dict = {}                  # stores the node names
gate_id_list = []                   # Stores the gate node numbers
gate_type_list = []                 # stores the type of each gate
gate_input_list = []                # stores the inputs of each gate as list of list
gate_cload_list = []                # stores the cload of each gate as list

input_filepath = './c17.bench'

# phase-2
str_data = ""
loadcap_inv = 0.0                   # cap of inv
gate_output_list = []               # stores the inputs of each gate as list of list
Q = deque()                         # for Top BFS
tempQ = []                          # phase-2 remove

# Class for gates:
class Node:
    def __init__(self,name):
        self.gate = ""          # gate type
        self.name = name        # gate name
        self.outpin = ""       # output node name
        self.cin = 0.0          # load capacitance
        self.cload = 0.0        # load capacitance
        self.inputs = []        # fanin details
        self.outputs = []       # fanout details
        self.visited = 0        # phase-2
        self.in_arr_time = {}
        self.tau_in = {}
        self.delay = {}
        self.pathslew = {}
        self.tau_out = 0.0
        self.out_arr_time = {}
        self.max_out_arr_time = 0.0
        self.required_time = {}
        self.min_required_time = 0.0
        self.slack = 0.0
    
        # phase-2
        # For storing cin
        gate_name = 'INV' if self.name.split('-')[0] == 'NOT' else 'BUF' if self.name.split('-')[0] == 'BUFF' else self.name.split('-')[0]
        for items in nodes.values():
            if gate_name == re.sub(r"\d", "", items.Allgate_name.split('_')[0]):
                self.cin = items.inputcap

# Function to write to a file:
def fn_w_circuit_file(is_truncate, file_name, mode, data):
    # is_truncate = 1 - truncates the specificed file "file_name" before writing "data"
    # mode = read/write/read-write
    with open(file_name, mode) as file:
        if is_truncate:
            file.truncate(0)
        file.write(data)

# Get input and output details:
def fn_io_parser(lines):
    # Parses the given line and identifies it as input, output or intermediate nodes
    # and adds them to corresponding dictionaries
    str_data = ""
    for line in lines:
        if line.strip() and not line.startswith('#'):
            line_detail = [s.strip() for s in re.split(r"(\()", line)]

            if (line_detail[0] == "INPUT"):
                name = line_detail[0].strip() + '-' + line_detail[2].split(")")[0]
                outpin = line_detail[2].split(")")[0]
                circuit_input_lines[outpin] = name
                gate = Node(name)
                gate.gate = name.split('-')[0]
                gate.in_arr_time[name] = 0.0
                gate.tau_in[name] = 0.002
                gate.outpin = outpin
                gate_obj_dict[name] = gate
            
            if (line_detail[0] == "OUTPUT"):
                name = line_detail[0].strip() + '-' + line_detail[2].split(")")[0]
                outpin = line_detail[2].split(")")[0]
                circuit_output_lines[outpin] = name
                gate = Node(name)
                gate.gate = name.split('-')[0]
                gate.outpin = outpin
                gate_obj_dict[name] = gate
            
            if (line_detail[0] not in "INPUT|OUTPUT"):
                line_detail = line_detail[0].split("=")
                circuit_intermediate_outputs[line_detail[0].strip()] = line_detail[1].strip() + '-' + line_detail[0].strip()

            str_data = (f"{len(circuit_input_lines)} primary inputs\n"
                        f"{len(circuit_output_lines)} primary outputs\n")
    
    # fn_w_circuit_file(1, circuitfile, 'a', str_data)

# Get gate count and create nodes:
def fn_gate_detail_parser(lines):
    # Responsible for parsing each line, identify the gate details and counts the distinct gates
    # creates objects for each gate/node, assigning all the necessary properties
    str_data = ""

    for line in lines:
        tempList = []
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
                
                for i in [s.strip() for s in line_detail[2].split("(")[1].split(")")[0].split(",")]:
                    if i in circuit_input_lines.keys():
                        tempList.append(circuit_input_lines.get(i))
                    elif i in circuit_intermediate_outputs.keys():
                        tempList.append(circuit_intermediate_outputs.get(i))
                    elif i in circuit_output_lines.keys():
                        tempList.append(circuit_output_lines.get(i))
                gate_input_list.append(tempList)

    # Create nodes
    for i in range(len(gate_id_list)):
        gate_name = gate_type_list[i] + '-' + gate_id_list[i]
        gate = Node(gate_name)
        gate_obj_dict[gate_name] = gate
        gate.gate = gate_type_list[i]
        gate.outpin = gate_id_list[i]
        gate.inputs = gate_input_list[i]

    # for key in gate_count_dict.keys():
        # str_data = str_data + f"{gate_count_dict[key]} {key} gates\n"

    # fn_w_circuit_file(0, circuitfile, 'a', str_data)

# Function for computing fan-in details of each gate/node:
def fn_fanin_parser():
    str_data = "\nFanin...\n"
    for index, gate in enumerate(gate_obj_dict):
        if (gate.split('-')[0] not in ['INPUT', 'OUTPUT']):
            str_data = str_data + gate + ':'
            for i in gate_obj_dict.get(gate).inputs:
                if (i.split('-')[1] in circuit_input_lines.keys()):
                    str_data = str_data + ' ' + circuit_input_lines.get(i.split('-')[1]) + ','
                    gate_obj_dict.get(gate).tau_in[i] = 0.002 # phase-2
                    gate_obj_dict.get(gate).in_arr_time[i] = 0.0 # phase-2
                elif (i.split('-')[1] in circuit_intermediate_outputs.keys()):
                    str_data = str_data + ' ' + circuit_intermediate_outputs.get(i.split('-')[1]) + ','
            str_data = str_data.strip(',') + '\n'
        
    # fn_w_circuit_file(0, circuitfile, 'a', str_data)

# Function for computing fan-out details for each gate/node:
def fn_fanout_parser():
    intermediate_key_list = list(circuit_intermediate_outputs.keys())
    str_data = "\nFanout...\n"

    for gate_id in gate_id_list:
        tempList = [] # phase-2
        cload = 0.0 # phase-2
        gatename = circuit_intermediate_outputs[gate_id]
        str_data = str_data + gatename + ':'
        if gate_id not in circuit_output_lines.keys():
            for index, input in enumerate(gate_input_list):
                input = [s.split('-')[1] for s in input]
                if gate_id in input:
                    id = intermediate_key_list[index]
                    gate_type = circuit_intermediate_outputs[id]
                    str_data = str_data + ' ' + gate_type + ','
                    tempList.append(gate_type) # phase-2
                    cload = cload + gate_obj_dict.get(gate_type).cin
            gate_obj_dict.get(gatename).cload = cload
        
        if gate_id in circuit_output_lines.keys():
           gate_type = circuit_output_lines[gate_id]
           str_data = str_data + ' ' + gate_type + ','
           tempList.append('OUTPUT-' + gate_id) # phase-2
           gate_obj_dict.get(gatename).cload = 4 * loadcap_inv

        str_data = str_data.strip(',') + '\n'
        gate_obj_dict.get(gatename).outputs = tempList # phase-2
    
    # fn_w_circuit_file(0, circuitfile, 'a', str_data)

# Function for reading the .bench file and populate the necessary circuit details:
def read_ckt():
    with open(input_filepath, "r") as file:
        data = file.readlines()
    global loadcap_inv
    loadcap_inv = nodes.get('INV_X1').inputcap
    fn_io_parser(data)
    fn_gate_detail_parser(data)
    fn_fanin_parser()
    fn_fanout_parser()