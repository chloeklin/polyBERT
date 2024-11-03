import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class SingleTask_Regressor(nn.Module):
    def __init__(self, input_dim: int,
                         output_dim: int = 1,
                         hidden_dim: int = 300):
        super(SingleTask_Regressor, self).__init__()
        
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.final = nn.Linear(hidden_dim, output_dim)

    def forward(self, x : torch.Tensor):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.final(x)
        return x
    



def train(modelname, model, optimizer, train_loader, val_loader, loss_fn, logging_dir, num_epochs=100):
    # Create TensorBoard logger
    writer = SummaryWriter(logging_dir)
    model_plotted = False

    # Early stopping parameters
    patience = 10
    best_loss = float('inf')
    trigger_times = 0
    
    # Set model to train mode
    model.train() 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training loop with early stopping
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            
            ## Step 1: Move input data to device (only strictly necessary if we use GPU)
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # For the very first batch, we visualize the computation graph in TensorBoard
            if not model_plotted:
                writer.add_graph(model, batch_inputs)
                model_plotted = True
            
            ## Step 2: Run the model on the input data
            preds = model(batch_inputs)
            preds = preds.squeeze() # Output is [Batch size, 1], but we want [Batch size]
            
            ## Step 3: Calculate the loss
            loss = loss_fn(preds, batch_targets.float())
            
            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero. 
            # The gradients would not be overwritten, but actually added to the existing ones.
            optimizer.zero_grad() 
            # Perform backpropagation
            loss.backward()
            
            ## Step 5: Update the parameters
            optimizer.step()
            
            ## Step 6: Take the running average of the loss
            epoch_loss += loss.item()
            
        # Add average loss to TensorBoard
        epoch_loss /= len(train_loader)
        writer.add_scalar('training_loss',
                          epoch_loss,
                          global_step = epoch + 1)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')
        
        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                val_outputs = model(batch_inputs)
                val_loss += loss_fn(val_outputs.squeeze(), batch_targets).item()
        val_loss /= len(val_loader)
        writer.add_scalar('val_loss',
                          val_loss,
                          global_step = epoch + 1)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    writer.close()
    state_dict = model.state_dict()
    torch.save(state_dict, modelname)

def test(model, test_loader, loss_fn):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            test_outputs = model(batch_inputs)
            test_loss += loss_fn(test_outputs.squeeze(), batch_targets).item()
    test_loss /= len(test_loader)
    print(f"MSE of the model: {test_loss:2f}")

@torch.no_grad() # Decorator, same effect as "with torch.no_grad(): ..." over the whole function.
def visualise_prediction(data, model, test_loader, loss_fn):
    model.eval()
    test_loss = 0.0
    target_pred = []
    target_true = []
    for batch_inputs, batch_targets in test_loader:
        test_outputs = model(batch_inputs)
        test_loss += loss_fn(test_outputs.squeeze(), batch_targets).item()
        target_pred.extend(test_outputs.squeeze().tolist())
        target_true.extend(batch_targets.tolist())
    
    test_loss /= len(test_loader)
    print(f"{data} Test MSE: {test_loss:2f}")
    target_true = np.array(target_true)
    target_pred = np.array(target_pred)
    
    fig = plt.figure(figsize=(4,4))
    plt.scatter(target_true, target_pred)
    plt.plot([target_true.min(), target_true.max()], [target_true.min(), target_true.max()], 'r--')
    plt.xlabel('True Test Labels')
    plt.ylabel('Predicted Test Labels')
    plt.title(f"{data}-{test_loss:2f}")
    plt.xticks([])
    plt.yticks([])

    return test_loss, fig