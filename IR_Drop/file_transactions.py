# Functions related to read/write to files

data = "" # Stores the file

# Read the input Spice Netlist:
def read_file(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    return data