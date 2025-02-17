VLSI Automation - Mini Project 1 - Static Timing Analysis:
The main objective of this project is to write a python file to parse a given liberty file and circuit file and compute the given circuit's characteristics.


Description:
The project consists of:
*.lib file - contains the properties of each gate i.e. load capacitance, delay and slew matrices.
*.bench file - contains the circuit/netlist details i.e. inputs, outputs and gates required for the circuit.
parser.py file reads the .lib and .bench files and outputs a file depending on the function called containing the requested details.
The following are the functions of interest:
--read_ckt - reads the *.bench file and returns a "circuit_details.txt" file containing the circuit's details. It contains the number of inputs, outputs, gates used along with fanin and fanout details of each gate.
--delay - reads the .lib file and returns a "delay_LUT.txt" file which contains the input delay, slew values along with the delay matrix for each gate.
--slew - reads the .lib file and returns a "slew_LUT.txt" file which contains the input delay, slew values along with the slew matrix for each gate.


Getting Started:
1. Install python version 3.7 or later.

2. Create a virtual environment for the project:
>> python3.7 -m venv <name of venv>

3. Activate the environment by sourcing it:
>> source <name of venv>/bin/activate

4. Install the required packages specified in the "requirements.txt" file:
>> pip3 install -r requirements.txt

5. Run the parser.py file:
syntax: python3.7 parser.py <(sub)function> <file>

Read the circuit details:
>> python3.7 parser.py --read_ckt c17.bench
>> Creates a circuit_details.txt in the same directory

Read the delay details from .lib (NLDM) file:
>> python3.7 parser.py --delay --read_nldm sample_NLDM.lib
>> Creates a delay_LUT.txt in the same directory

Read the slew details from .lib (NLDM) file:
>> python3.7 parser.py --slew --read_nldm sample_NLDM.lib
>> Creates a slew_LUT.txt in the same directory


An original work of:
Copyright © 2025 by Mahathi Krishna Ravi Shankar and Vishnu Ram Jawaharram. All rights reserved.