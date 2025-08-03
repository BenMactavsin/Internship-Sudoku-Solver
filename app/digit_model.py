import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Neural Network
class DigitModel(nn.Module):
    def __init__(self):
        super(DigitModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), # Flatten 28x28 input to 784
            nn.Linear(784, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    epochs = 1000
    model_path = "model/mnist_model.pt"

    # Device Configuration (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()  # Function for converting image to tensor with values [0,1]

    # Download and load training data
    train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Load the model and move it to the selected device
    model = DigitModel().to(device)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss() # For multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    print("Training started...\n")
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss}")

    print("\nTraining complete!")

    # Evaluation on Test Set
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")

    # Save the Trained Model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")
