import subprocess
import os

directory = "./Training_Data"
script_path = "./ir_solver.py"  

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):  # Ensure it's a file
        result = subprocess.run(['python', script_path, file_path], capture_output=True, text=True)
        
        # print(f"Processing {filename}")
        print("Output:\n", result.stdout)
        # if result.stderr: