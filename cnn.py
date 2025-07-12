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
        image = torch.from_numpy(np.array(pixels).reshape(1, width, height))
        label = self.img_labels.iloc[idx, 1].reshape(1)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        #print("the shape of label is:", label.shape)
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


def get_data_loader(dir, make_annotations_file=True, batch_size=64):
    img_dir = f"saved_screens/{dir}"
    annotation_file_path = f"{dir}_annotations"
    if make_annotations_file:
        make_dataset_annotations_file(img_dir, annotation_file_path)
    dataset = CustomImageDataset(annotation_file_path, img_dir, lambda img : torch.flip(img, [0]).to(torch.float), lambda lbl : torch.tensor(lbl).to(torch.float))
    return DataLoader(dataset, batch_size=batch_size)


def test_custom_dataset():
    img_dir = "saved_screens/test_dataset"
    annotations_file = "test_an_file"
    transform = lambda img : torch.flip(img, [0])
    make_dataset_annotations_file(img_dir, annotations_file)
    cids = CustomImageDataset(annotations_file, img_dir, transform)

    idx = 0
    img, _ = cids[idx]
    x, y, z, r = cids.get_all_labels(idx)
    label = f"pos = ({round(x)}, {round(y)}, {round(z)}), rad = {round(r)}"
    show_my_tensor(img, label)

def new_test(dir):
    img_dir = f"saved_screens/{dir}"
    annotation_file_path = f"{dir}_annotations"
    dataset = CustomImageDataset(annotation_file_path, img_dir, lambda img : torch.flip(img, [0]).to(torch.float), lambda lbl : torch.tensor(lbl).to(torch.float))
    print(dataset[0])


'''
old cnn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, 2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2) # 126 * 126
        self.fc1 = nn.Linear(4 * 62 * 62, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
'''


# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, 4)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(3844, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def test_example(model, test_data_dir):
    test_dataloader = get_data_loader(test_data_dir, make_annotations_file=True, batch_size=1)
    img, lbl = next(iter(test_dataloader))
    output = model(img)
    print(f"Predicted value: {output}")
    print(f"True value: {lbl}")
    print(f"(error = {output - lbl})")



def run_cnn():    
    train_loader = get_data_loader("train_data", make_annotations_file=False)
    test_loader = get_data_loader("test_data", make_annotations_file=False)

    # Set up TensorBoard writer
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/CNN"
    writer = SummaryWriter(log_dir)

    # Instantiate the model (using cuda if it is available)
    model = CNN().cuda() if torch.cuda.is_available() else CNN()

    # Define optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    #loss_fn = nn.MSELoss(reduction='none')
    loss_fn = nn.MSELoss()

    # Training loop
    num_epochs = 5  # Vary as you wish
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total = 0

        all_my_running_losses = []

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
            total += labels.size(0)

            all_my_running_losses.append(running_loss)

        print(all_my_running_losses[-1])
        print(running_loss / len(train_loader))
        
        # Log loss and accuracy to TensorBoard
        writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch)

        # Validation loop
        model.eval()
        val_loss = 0.0
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
                val_total += labels.size(0)

        # Log validation loss
        writer.add_scalar('Loss/val', val_loss / len(test_loader), epoch)

    # close the writer
    writer.close()
    print("--------------")

    # Summarise the model
    summary(model, input_size=(1, 256, 256))


    #test_example(model, "temp_test_folder")

run_cnn()

# currently predicting second column (column 1) which should be the x value