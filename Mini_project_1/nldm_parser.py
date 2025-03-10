import re           #To parse throug the NLDM or the Circuit file 
import numpy as np  #To handle arrays

# Parameters
nodes = []                          #To hold all the nodes of the LUT class
Allgate_name = ''                   #To get all the gate names from the NLDM file
All_delay = ''                      #To hold all the dleay values from the NLDM file
All_slews=''                        #To hold slew data from the NLDM file
Cload_vals=''                       #To holde the load cap values from the nldm file
Tau_in_vals = ''                    #To hold slew data from the NLDM file
circuitfile = 'ckt_details.txt'     #Output file for the bench parser
data = []                           # holds the file data
capacitance = []
inputcap = 0

input_filepath = './sample_NLDM.lib'

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

# phase-2 start:
def get_delay(gate_name,Cload,Tau):
    tauidx = 0
    loadidx = 0
    idx0 = idx1 = idx2 = idx3 = t1 = t2 = c1 = c2 = 0
    v = 0.0
    gate_name = 'INV' if gate_name.split('-')[0] == 'NOT' else 'BUF' if gate_name.split('-')[0] == 'BUFF' else gate_name.split('-')[0]
    for nod in nodes:
        if re.sub(r"\d", "", nod.Allgate_name.split('_')[0]) == gate_name:
            Cload_list = [float(num) for num in nod.Cload_vals.split(",") if num.strip()]
            Tauin_list = [float(num) for num in nod.Tau_in_vals.split(",") if num.strip()]
            #print(Cload_list,Tauin_list)
            c = Cload
            t = Tau
            for index,val in enumerate(Cload_list):
                if val == Cload:
                    loadidx = index
            for index,val in enumerate(Tauin_list):
                if val == Tau:
                    tauidx = index
            
            if (loadidx == 0) and (tauidx == 0):
                idx0,idx1,idx2,idx3,t1,t2,c1,c2= interpolate(Cload,Tau,Cload_list,Tauin_list)
                print(c1,c2,t1,t2)
                print(c,t)
                v11 = float(nod.All_delays[idx2][idx0])
                v12 = float(nod.All_delays[idx2][idx1])
                v21 = float(nod.All_delays[idx3][idx0])
                v22 = float(nod.All_delays[idx3][idx1])
                print(v11,v12,v21,v22)
                v = v11*(c2-c)*(t2-t)+v12*(c-c1)*(t2-t)+v21*(c2-c)*(t-t1)+v22*(c-c1)*(t-t1)/(c2-c1)*(t2-t1) 
            else:
                v = (tauidx,loadidx)
    print(v) 
    #print(nod.All_delays[loadidx][tauidx])

def interpolate(Cload,Tau,Cload_list,Tauin_list):
    t1 = t2 = c1 = c2 = 0
    for idx_cap,cap in enumerate(Cload_list):
        if cap<=Cload:
            c1 = cap
            idx0 = idx_cap
        if cap>=Cload:
            c2 = cap
            idx1 = idx_cap
            break   
    for idx_slew,slew in enumerate(Tauin_list):
        if slew<=Tau:
            t1 = slew
            idx2 = idx_slew
        if slew>=Tau:
            t2 = slew
            idx3 = idx_slew
            break
    return(idx0,idx1,idx2,idx3,t1,t2,c1,c2)
# phase-2 end