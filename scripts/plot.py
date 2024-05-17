import pandas as pd
import matplotlib.pyplot as plt 

## AdamW Optimizer
# Load the CSV file
file_path = '/ccs/home/adityatomar/improved-diffusion/logs/adamw/progress.csv'
data = pd.read_csv(file_path)

# Extract the relevant columns
STEP = 50
loss = data['loss'][::STEP]
step = data['step'][::STEP]

# Plot loss vs step
plt.figure(figsize=(10, 6)) 
plt.plot(step, loss, marker='o', linestyle='-', label="AdamW")

## JorgeKFAC Optimizer
#file_path = '/ccs/home/adityatomar/improved-diffusion/logs/jorge/progress.csv'
#data = pd.read_csv(file_path)
#
## Extract the relevant columns
#loss = data['loss'][::STEP]
#step = data['step'][::STEP]
#
## Plot loss vs step
#plt.figure(figsize=(10, 6)) 
#plt.plot(step, loss, marker='o', linestyle='-', label="Jorge")

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss vs Step')
plt.grid(True)
save_path = '/ccs/home/adityatomar/improved-diffusion/output.png'
plt.savefig(save_path)
