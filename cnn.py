import torch
import torch.nn as nn   # Building blocks for neural networks
import torch.nn.functional as F # Various functions for building neural networks
import torch.optim as optim  # Optimiser for neural networks
from torchsummary import summary  # Summarise PyTorch model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter # Tensorboard in PyTorch
import datetime
#!rm -rf ./logs/
print(torch.__version__) # Double check the colab has the instance of tensorflow we want

# If using GPU
'''print('Cuda Available : {}'.format(torch.cuda.is_available()))
if torch.cuda.is_available():
  print('GPU - {0}'.format(torch.cuda.get_device_name()))'''

# Library for Progres Bar during training
#!pip install tqdm
#!pip install tensorboard

from tqdm import tqdm # Progress bar during training




import os
import pandas as pd
import numpy as np
import graphics.graphics_utils as mn

import matplotlib.pyplot as plt

def make_dataset_annotations_file(dir, dest_path):
    df = pd.DataFrame(columns=["img_name","x","y","z","r"])
    df["img_name"] = os.listdir(dir)
    for idx, i in enumerate(os.listdir(dir)):
        _, _, x, y, z, r, _  = mn.load_screen(f"{dir}/{i}")
        """df["x"][idx] = x
        df["y"][idx] = y
        df["z"][idx] = z
        df["r"][idx] = r"""

        df.loc[idx, "x"] = x
        df.loc[idx, "y"] = y
        df.loc[idx, "z"] = z
        df.loc[idx, "r"] = r



    df.to_csv(dest_path, index=False, header=True)


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #image = decode_image(img_path)
        img_data = mn.load_screen(img_path)
        width, height = img_data[0], img_data[1]
        pixels = img_data[6]
        image = torch.from_numpy(np.array(pixels).reshape(width, height))
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def get_all_labels(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        _, _, x, y, z, r, _ = mn.load_screen(img_path)
        return [x, y, z, r]
    

def show_my_tensor(img, label):
    plt.figure()
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def test_it():
    img_dir = "saved_screens/test_dataset"
    an_file = "test_an_file"
    transform = lambda img : torch.flip(img, [0])
    make_dataset_annotations_file(img_dir, an_file)
    cids = CustomImageDataset(an_file, img_dir, transform)

    idx = 0
    img, _ = cids[idx]
    x, y, z, r = cids.get_all_labels(idx)
    label = f"pos = ({round(x)}, {round(y)}, {round(z)}), rad = {round(r)}"
    show_my_tensor(img, label)

test_it()

"""

# start of MLP

class MLP(nn.Module):
  def __init__(self):
      super(MLP, self).__init__()
      self.model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
        nn.Softmax(dim=1)
    )

  def forward(self, x):
      x = self.model(x)
      return x

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize the pixel values
])

# Load MNIST training dataset with transformations
mnist_train = datasets.MNIST('data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('data', train=False, download=True, transform=transform)


# Split dataset into training, validation, and test sets
train_size = int(0.8 * len(mnist_train))
val_size = len(mnist_train) - train_size
mnist_train, mnist_val = random_split(mnist_train, [train_size, val_size])

# Define data loaders
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=64)
test_loader = DataLoader(mnist_test, batch_size=64)





'''class MLP(nn.Module):
  def __init__(self):
      super(MLP, self).__init__()
      self.flatten = nn.Flatten()
      self.fc1 = nn.Linear(28*28, 512)
      self.fc2 = nn.Linear(512, 10)

  def forward(self, x):
      x = self.flatten(x)
      x = F.relu(self.fc1(x))
      x = F.softmax(self.fc2(x), dim=1)
      return x'''
  
# Instantiate MLP
model = MLP().cuda() if torch.cuda.is_available() else MLP()

# Optimiser
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Set up TensorBoard log directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/MLP"
writer = SummaryWriter(log_dir)

# Example training loop
num_epochs = 5 # Vary as you wish


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        # When cuda is available
        if torch.cuda.is_available():
          inputs = inputs.cuda()
          labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # Logging training loss and accuracy
    writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
    writer.add_scalar('Accuracy/train', 100. * correct / total, epoch)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:  # assume you have DataLoader for test_ds
            # When cuda is available
            if torch.cuda.is_available():
              inputs = inputs.cuda()
              labels = labels.cuda()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    # Logging validation loss and accuracy
    writer.add_scalar('Loss/val', val_loss / len(val_loader), epoch)
    writer.add_scalar('Accuracy/val', 100. * val_correct / val_total, epoch)

# Don't forget to close the writer when done
writer.close()
print("--------------")
# Summarise the model
summary(model, input_size=(1, 28*28))
     


'''%load_ext tensorboard
%tensorboard --logdir ./logs/fit'''



# Set up TensorBoard writer
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/CNN"
writer = SummaryWriter(log_dir)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # MNIST has 1 color (channel)
        self.fc1 = nn.Linear(32 * 26 * 26, 512)  # Flatten after Conv2D
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model (using cuda if it is available)
model = CNN().cuda() if torch.cuda.is_available() else CNN()

# Define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Training loop
num_epochs = 5  # Vary as you wish
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        # When cuda is available
        if torch.cuda.is_available():
          inputs = inputs.cuda()
          labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # Log loss and accuracy to TensorBoard
    writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)
    writer.add_scalar('Accuracy/train', 100. * correct / total, epoch)

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # When cuda is available
            if torch.cuda.is_available():
              inputs = inputs.cuda()
              labels = labels.cuda()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    # Log validation loss and accuracy
    writer.add_scalar('Loss/val', val_loss / len(test_loader), epoch)
    writer.add_scalar('Accuracy/val', 100. * val_correct / val_total, epoch)

# Don't forget to close the writer
writer.close()
print("--------------")

# Summarise the model
summary(model, input_size=(1, 28, 28))


"""