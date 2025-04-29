# Function for reading the files:
data = "" # Stores the file

# Read the input Spice Netlist:
def Read_File(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    return data