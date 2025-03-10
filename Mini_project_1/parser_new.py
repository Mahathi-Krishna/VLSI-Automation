import argparse     #For Command line interfacing
import re           #To parse throug the NLDM or the Circuit file 
import numpy as np  #To handle arrays

# phase-2
from collections import deque

# parser = argparse.ArgumentParser() 
# parser.add_argument("-d", "--delay", help="print delay",action="store_true")                #To run for delay using the Command Line Interface
# parser.add_argument("-s", "--slew", help="print slew",action="store_true")                  #To parse and populate the datastructure for the NLDM file and find delay
# parser.add_argument("-r","--read_nldm",help="reads  the nldm file",action = "store_true")   #To parse and populate the datastructure for the NLDM file and find slew
# parser.add_argument("-c","--read_ckt",help="reads the bench file",action="store_true")      #To parse and populate the datastructure for the becnch file
# parser.add_argument('input_file',help='Path to the input file')                             # Mentioning the input file which needs to be parsed

# args = parser.parse_args()
# input_filepath = args.input_file    # Getting the file mentioned in the command line

# Parameters
nodes = []                          #To hold all the nodes of the LUT class
Allgate_name = ''                   #To get all the gate names from the NLDM file
All_delay = ''                      #To hold all the dleay values from the NLDM file
All_slews=''                        #To hold slew data from the NLDM file
Cload_vals=''                       #To holde the load cap values from the nldm file
Tau_in_vals = ''                    #To hold slew data from the NLDM file
circuitfile = 'ckt_details.txt'     #Output file for the bench parser
data = []                           # holds the file data
circuit_input_lines = {}            # holds the input line and type details
circuit_intermediate_outputs = {}   # holds the intermediate line details
circuit_output_lines = {}           # holds the output line details
gate_count_dict = {}                # holds the count for each gate type
gate_obj_list = []                  # stores the node names
gate_obj_dict = {}                  # stores the node names
gate_id_list = []                   # Stores the gate node numbers
gate_type_list = []                 # stores the type of each gate
gate_input_list = []                # stores the inputs of each gate as list of list
gate_cload_list = []                # stores the cload of each gate as list
capacitance = []
inputcap = 0

# phase-2
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
        self.tau_in = []
        self.tau_out = 0.0
        self.in_arr_time = []
        self.out_arr_time = []
        self.max_out_arr_time = 0.0

# Class for NLDM:
class LUT:
    def __init__(self,Allgate_name,All_delay,All_slews,Cload_vals,Tau_in_vals,inputcap):
        self.Allgate_name = Allgate_name
        self.All_delays = All_delay
        self.All_slews = All_slews
        self.Cload_vals = Cload_vals
        self.Tau_in_vals = Tau_in_vals
        self.inputcap = inputcap
    def assign_arrays(self,NLDM_file):  #Function to pass the NLDM file and retrive the data , returns the required data from the NLDM file
        nodes = []
        lines1 = []
        gate_index = []
        gates_nldm = []
        input_slew = []
        load_cap = []
        values = []
        all_values = []
        flag = 0
        id = 0
        value_str=""
        cap = []
        
        with open(NLDM_file,"r") as file:
            for line in file:
                cleaned_line = line.strip()
                lines1.append(cleaned_line)
        
        for i in range(0,len(lines1)):
            if(("cell" in lines1[i]) and ("cell_delay" not in lines1[i])):
                gate_index.append(i)
                inputs = re.split(r"cell ", lines1[i])
                gates = inputs[1]
                gates = re.split(r"[(.*?)]", str(inputs))[1]
                gates_nldm.append(gates)
            if(('capacitance		:' in lines1[i])):
                cap.append(float(lines1[i].split(':')[1].strip(';')))
            if("index_1" in lines1[i]):
                input_slew.append(lines1[i].split('"')[1].strip())
            if("index_2" in lines1[i]):
                load_cap.append(lines1[i].split('"')[1].strip())
    
            if("values (" in lines1[i]):
                id = i
                flag = 1
            if(((flag==1) and(i>=id)) and (");" in str(lines1[i]))):
                value_str = value_str+str(lines1[i])

                values.append(value_str)
                id = 0
                flag = 0
                value_str = ""
            elif(i >= id and flag == 1):
                value_str = value_str+str(lines1[i])
        
        for i in range(0,len(values)):
            values1 = values[i].split('(')[1:][0]
            values1 = [(value.strip().replace('"', '').replace("\\", "").replace(");","")) for value in values1.split(",")]
            values1 = np.array(values1).reshape(7,7)
            all_values.append(values1)
        return(gates_nldm,input_slew,load_cap,all_values,cap)

# Funciton called when the command line calls to parse for nldm file
def nldm():
    lut_instance = LUT(Allgate_name,All_delay,All_slews,Cload_vals,Tau_in_vals,inputcap)
    gates_nldm,input_slew,load_cap,all_values,capacitance = lut_instance.assign_arrays(input_filepath) # phase-2
    for i in range(0,len(gates_nldm)):
        if(i==0):
            node = LUT(gates_nldm[i],all_values[i],all_values[i+1],load_cap[i+2],input_slew[i+2],capacitance[i])
            nodes.append(node)
        else:
            node = LUT(gates_nldm[i],all_values[i+i],all_values[i+i+1],load_cap[i+2],input_slew[i+2],capacitance[i])
            nodes.append(node)
    
    # phase-2
    for index, gate in enumerate(gate_obj_list):    
        # For storing cin
        gate_name = 'INV' if gate.name.split('-')[0] == 'NOT' else 'BUF' if gate.name.split('-')[0] == 'BUFF' else gate.name.split('-')[0]
        for items in nodes:
            if gate_name == re.sub(r"\d", "", items.Allgate_name.split('_')[0]):
                gate.cin = items.inputcap

# Function to call for delay
def delay():
    f = open("delay_LUT.txt","w")
    for node in nodes:
        f.write("cell: "+ node.Allgate_name+"\n")
        f.write("input slews: "+ node.Tau_in_vals+"\n")
        f.write("load_cap: "+ node.Cload_vals+"\n\n")
        f.write("delays:\n")
        for row in node.All_delays:
            temp = (' '.join(row)+";\n\n")
            f.write(temp)

# Function to call for slew
def slew():
    f = open("slew_LUT.txt","w")
    for node in nodes:
        f.write("cell: "+ node.Allgate_name+"\n")
        f.write("input slews: "+ node.Tau_in_vals+"\n")
        f.write("load_cap: "+ node.Cload_vals+"\n\n")
        f.write("slews:\n")
        for row in node.All_slews:
            temp = (' '.join(row)+";\n\n")
            f.write(temp)

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
                circuit_input_lines[line_detail[2].split(")")[0]] = line_detail[0].strip() + '-' + line_detail[2].split(")")[0]
            
            if (line_detail[0] == "OUTPUT"):
                circuit_output_lines[line_detail[2].split(")")[0]] = line_detail[0].strip() + '-' + line_detail[2].split(")")[0]
            
            if (line_detail[0] not in "INPUT|OUTPUT"):
                line_detail = line_detail[0].split("=")
                circuit_intermediate_outputs[line_detail[0].strip()] = line_detail[1].strip() + '-' + line_detail[0].strip()

            str_data = (f"{len(circuit_input_lines)} primary inputs\n"
                        f"{len(circuit_output_lines)} primary outputs\n")
    
    fn_w_circuit_file(1, circuitfile, 'a', str_data)

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
        gate_obj_list.append(gate)
        gate_obj_dict[gate_name] = gate
        gate.gate = gate_type_list[i]
        gate.outpin = gate_id_list[i]
        gate.inputs = gate_input_list[i]

    for key in gate_count_dict.keys():
        str_data = str_data + f"{gate_count_dict[key]} {key} gates\n"

    fn_w_circuit_file(0, circuitfile, 'a', str_data)

# Function for computing fan-in details of each gate/node:
def fn_fanin_parser():
    str_data = "\nFanin...\n"
    for index, gate in enumerate(gate_obj_dict):
        str_data = str_data + gate + ':'
        for i in gate_obj_dict.get(gate).inputs:
            if (i.split('-')[1] in circuit_input_lines.keys()):
                str_data = str_data + ' ' + circuit_input_lines.get(i.split('-')[1]) + ','
                gate_obj_dict.get(gate).tau_in.append(0.002) # phase-2
                gate_obj_dict.get(gate).in_arr_time.append(0) # phase-2
            elif (i.split('-')[1] in circuit_intermediate_outputs.keys()):
                str_data = str_data + ' ' + circuit_intermediate_outputs.get(i.split('-')[1]) + ','
        str_data = str_data.strip(',') + '\n'
        
    fn_w_circuit_file(0, circuitfile, 'a', str_data)

# Function for computing fan-out details for each gate/node:
def fn_fanout_parser():
    intermediate_key_list = list(circuit_intermediate_outputs.keys())
    str_data = "\nFanout...\n"

    # phase-2
    for node in nodes:
        if(node.Allgate_name == 'INV_X1'):
            loadcap_inv = node.inputcap
    
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
           tempList.append('O') # phase-2
           gate_obj_dict.get(gatename).cload = 4 * loadcap_inv

        str_data = str_data.strip(',') + '\n'
        gate_obj_dict.get(gatename).outputs = tempList # phase-2
    
    str_data = str_data + '\n\n'
    for i in gate_obj_list:
        mystr = i.__dict__
        str_data = str_data + str(mystr) + '\n'
    fn_w_circuit_file(0, circuitfile, 'a', str_data)

# Function for reading the .bench file and populate the necessary circuit details:
def read_ckt():
    with open(input_filepath, "r") as file:
        data = file.readlines()
    fn_io_parser(data)
    fn_gate_detail_parser(data)
    # fn_fanout_parser()
    fn_fanin_parser()


# if args.read_nldm:
#     nldm()
#     if args.delay:
#         delay()
#     if args.slew:
#         slew()
# if args.read_ckt:
#     read_ckt()

input_filepath = './c17.bench'
read_ckt()
input_filepath = './sample_NLDM.lib'
nldm()
delay()
slew()
fn_fanout_parser()

# print("Enter:\n1. Read Ckt\n2. Read NLDM")
# opt = int(input())

# if opt==2:
#     input_filepath = './sample_NLDM.lib'
#     print("Enter:\n1. Delay\n2. Slew")
#     opt2 = int(input())
#     nldm()
#     if opt2==1:
#         delay()
#     elif opt2==2:
#         slew()
# elif opt==1:
#     input_filepath = './c17.bench'
#     read_ckt()

# phase-2 start
# For topological traversal
# def topsort(v):
#     for index, gate in enumerate(gate_obj_list):
#         if (v in gate.inputs) and (gate.visited == 0) :
#             gate.visited = 1
#             Q.append(gate.outname)
#             tempQ.append(gate.outname)

# for i in circuit_input_lines:
#     Q.append(i)
#     tempQ.append(i)

# while Q:
#     v = Q.popleft()
#     topsort(v)

# print(tempQ)

# for i in gate_obj_list:
#     print(i.name, i.inputs, i.cin, i.outputs, i.cload)
#    print(i.__dict__)
# phase-2 end