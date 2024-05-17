import pandas as pd
import matplotlib.pyplot as plt 

# Load the CSV file
file_path = '/ccs/home/adityatomar/improved-diffusion/logs/progress.csv'
data = pd.read_csv(file_path)

# Extract the relevant columns
loss = data['loss']
step = data['step']

# Plot loss vs step
plt.figure(figsize=(10, 6)) 
plt.plot(step, loss, marker='o', linestyle='-')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss vs Step')
plt.grid(True)
save_path = '/ccs/home/adityatomar/improved-diffusion/output.png'
plt.savefig(save_path)
