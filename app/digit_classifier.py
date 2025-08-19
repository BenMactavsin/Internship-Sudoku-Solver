from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from cv2.typing import MatLike

# Nöral Ağ (LeNet 5)
class DigitClassifier(nn.Module):
    def __init__(self: "DigitClassifier"):
        super(DigitClassifier, self).__init__()
        self.layers = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2),
          nn.Sigmoid(),
          nn.MaxPool2d(kernel_size=(2, 2), stride=2),
          nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), padding=0),
          nn.Sigmoid(),
          nn.MaxPool2d(kernel_size=(2, 2), stride=2),
          nn.Flatten(),
          nn.Linear(in_features=400, out_features=120),
          nn.Sigmoid(),
          nn.Linear(in_features=120, out_features=84),
          nn.Sigmoid(),
          nn.Linear(in_features=84, out_features=10)
        )

    def forward(self: "DigitClassifier", x):
        return self.layers(x)
    
    def load_from(self: "DigitClassifier", path: Path) -> int:
        file = torch.load(str(path.resolve()), map_location=torch.device("cpu"))
        self.load_state_dict(file['model'])
    
    def classify_digit(self: "DigitClassifier", image: MatLike) -> int:
        return self(transforms.ToTensor()(image).unsqueeze(0)).argmax()