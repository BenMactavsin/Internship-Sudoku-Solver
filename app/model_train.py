import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from digit_classifier import DigitClassifier

if __name__ == "__main__":
  if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount(r"/mnt/drive")
    root_folder = Path(r"/mnt/drive/MyDrive/mnist-model-root")
  else:
    root_folder = Path(r"G:\My Drive\mnist-model-root")

  # Hiperparametreler
  batch_size = 256
  learning_rate = 0.0001
  epochs = 400

  checkpoint_folder = root_folder/"checkpoint"
  data_folder = root_folder/"data"
  checkpoint_file_path = checkpoint_folder/"396.pth"

  checkpoint_folder.mkdir(parents=True, exist_ok=True)
  data_folder.mkdir(parents=True, exist_ok=True)

  # Device Configuration (GPU if available)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  transform = transforms.ToTensor()  # Function for converting image to tensor with values [0,1]

  # Download and load training data
  train_dataset = datasets.QMNIST(root=data_folder, what="train", download=True, transform=transform)
  test_dataset  = datasets.QMNIST(root=data_folder, what="test", download=True, transform=transform)

  # Create data loaders
  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  start_epoch = 0
  max_accuracy = 0
  model = DigitClassifier().to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  epoch_history = []
  if (checkpoint_file_path.exists()):
    checkpoint_file = torch.load(checkpoint_file_path, map_location=device)
    start_epoch = checkpoint_file['epoch']
    model.load_state_dict(checkpoint_file['model'])
    criterion.load_state_dict(checkpoint_file['criterion'])
    optimizer.load_state_dict(checkpoint_file['optimizer'])
    epoch_history = checkpoint_file['epoch-history']
    max_accuracy = sorted(epoch_history, key=lambda x: x[2], reverse=True)[0][2]

  for v in epoch_history:
    print(f"Epoch [{v[0]}/{epochs}] - Loss: {v[1]} - Accuracy: {v[2]}")

  # Training Loop
  print("Training started...\n")
  for epoch in range(start_epoch + 1, epochs + 1):
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
    epoch_history.append((epoch, avg_loss, accuracy))
    print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss} - Accuracy: {accuracy}")

    if (accuracy > max_accuracy):
      max_accuracy = accuracy
      checkpoint_auto_path = checkpoint_folder / f'{epoch}.pth'
      torch.save({
          'epoch': epoch,
          'model': model.state_dict(),
          'criterion': criterion.state_dict(),
          'optimizer': optimizer.state_dict(),
          'epoch-history': epoch_history
      }, checkpoint_auto_path)
      print(f"Checkpoint saved to '{checkpoint_auto_path}'")

  print("\nTraining complete!")