import json
import matplotlib.pyplot as plt
import os

parent_directory = 'MDM_Resnet50_RNN_25_epochs_1_location_trained_20240318-111308'
batch_losses_file_path = f'{parent_directory}/batch_losses.json'
epoch_losses_file_path = f'{parent_directory}/epoch_losses.json'
val_epoch_losses_file_path = f'{parent_directory}/val_epoch_losses.json'

########## BATCHES ##########

# Load the loss values from the JSON file
with open(batch_losses_file_path, 'r') as f:
    loss_values = json.load(f)

# Create a line plot of the loss values
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.plot(loss_values, label='Loss', color='blue')  # You can customize the color and label
plt.title('Model Loss Over Batches')  # Set the title of the plot
plt.xlabel('Batch')  # Set the label for the x-axis
plt.ylabel('Loss')  # Set the label for the y-axis
plt.legend()  # Show legend
plt.grid(True)  # Show grid
# plt.ylim(0, 1.5)  # Set the lower and upper y-axis limits
plt.yscale('log')

# Save the plot in the specified directory
plot_file_name = 'batch_loss_plot.png'  # Name of the plot file
plot_path = os.path.join(parent_directory, plot_file_name)
plt.savefig(plot_path)

print(f"Plot saved to {plot_path}")

########## EPOCHS ##########

# Load the loss values from the JSON file
with open(epoch_losses_file_path, 'r') as f:
    loss_values = json.load(f)

with open(val_epoch_losses_file_path, 'r') as f:
    val_loss_values = json.load(f)

# Create a line plot of the loss values
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.plot(loss_values, label='Loss', color='blue')  # You can customize the color and label
plt.plot(val_loss_values, label='Validation Loss', color='red')  # Plot validation loss
plt.title('Model Loss Over Epochs')  # Set the title of the plot
plt.xlabel('Epoch')  # Set the label for the x-axis
plt.ylabel('Loss')  # Set the label for the y-axis
plt.legend()  # Show legend
plt.grid(True)  # Show grid
# plt.ylim(0, 1.5)  # Set the lower and upper y-axis limits
plt.yscale('log')

# Save the plot in the specified directory
plot_file_name = 'epoch_loss_plot.png'  # Name of the plot file
plot_path = os.path.join(parent_directory, plot_file_name)
plt.savefig(plot_path)

print(f"Plot saved to {plot_path}")