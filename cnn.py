import torch
import torch.nn as nn   # Building blocks for neural networks
import torch.nn.functional as F # Various functions for building neural networks
import torch.optim as optim  # Optimiser for neural networks
from torchsummary import summary  # Summarise PyTorch model
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter # Tensorboard in PyTorch
import datetime


from tqdm import tqdm # Progress bar during training


import os
import pandas as pd
import numpy as np
import graphics.graphics_utils as mn

import matplotlib.pyplot as plt

DATA_PATH = "./cnn_data"

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
        img_data = mn.load_screen(img_path)
        width, height = img_data[0], img_data[1]
        pixels = img_data[6]
        image = torch.from_numpy(np.flip(np.array(pixels).reshape(1, width, height), 1).copy())
        label = self.img_labels.iloc[idx, 1:5]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
def get_mean_and_std(data_loader):
    total_mean = 0
    total_var = 0
    num_imgs = 0
    for images, _ in data_loader:
        num_imgs += images.size(0)

        images = images.view(images.size(0), images.size(1), -1)
        total_mean += images.mean(2).sum(0)
        total_var += images.var(2).sum(0)

    mean = total_mean / num_imgs
    std = torch.sqrt(total_var / num_imgs)
    return (mean, std)

def show_my_tensor(img, title):
    invTrans = transforms.Compose([transforms.Normalize((0.0,), (1/39.3379,)),
                                   transforms.Normalize((-197.6725), (1.0,))])
    img = invTrans(img)
    plt.figure()
    plt.title(title)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=255)

def scale_labels(label):
    """
    Converts all values to be between -0.5 and 0.5
    """
    x, y, z, r = label
    scaled_x = x / 400.0
    scaled_y = (y - 300.0) / 400.0
    scaled_z = z / 400.0
    scaled_r = (r - 50.0) / 50.0
    return torch.tensor([scaled_x, scaled_y, scaled_z, scaled_r]).to(torch.float)

def inv_scale_labels(label):
    """
    Converts values back to their original units
    """
    scaled_x, scaled_y, scaled_z, scaled_r = label
    x = scaled_x * 400.0
    y = scaled_y * 400.0 + 300.0
    z = scaled_z * 400.0
    r = scaled_r * 50.0 + 50.0
    return [x, y, z, r]


def get_data_loader(img_dir, batch_size=64):
    annotations_file_path = f"{img_dir}_annotations"
    if not os.path.exists(annotations_file_path):
        make_dataset_annotations_file(img_dir, annotations_file_path)
    transform = lambda img : nn.AvgPool2d(kernel_size=4, stride=4)(transforms.Normalize((197.6725,), (39.3379,))(img.to(torch.float)))
    target_transform = scale_labels
    dataset = CustomImageDataset(annotations_file_path, img_dir, transform, target_transform)
    return DataLoader(dataset, batch_size=batch_size)



# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(1800, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def show_prediction(model, img, lbls):
    pred_lbls = inv_scale_labels(model(img)[0])
    true_lbls = inv_scale_labels(lbls[0])

    pred_lbls = list(map(lambda x: round(float(x), 4), pred_lbls))
    true_lbls = list(map(lambda x: round(float(x), 4), true_lbls))

    title = f"True: {true_lbls}\nPredicted: {pred_lbls}"
    show_my_tensor(img, title)



def run_cnn(train_dir, test_dir, save_to="", num_epochs=5):
    train_loader = get_data_loader(train_dir)
    test_loader = get_data_loader(test_dir)

    # Set up TensorBoard writer
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/CNN"
    writer = SummaryWriter(log_dir)

    # Instantiate the model (using cuda if it is available)
    model = CNN().cuda() if torch.cuda.is_available() else CNN()

    # Define optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    loss_fn = nn.MSELoss()

    # Training loop
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

            total += labels.size(0)

            running_loss += loss.item()
            all_my_running_losses.append(running_loss)

        
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

    if save_to:
        # save model parameters
        torch.save(model.state_dict(), save_to)

    # Summarise the model
    summary(model, input_size=(1, 64, 64))


def load_cnn(file_path):
    # load model parameters from given file
    model = CNN().cuda() if torch.cuda.is_available() else CNN()
    model.load_state_dict(torch.load(file_path))
    return model


def test_examples(model_file: str, img_dir: str):
    """
    Show the predictions for each image in img_dir
    """
    model = load_cnn(model_file)
    loader = get_data_loader(img_dir, batch_size=1)
    for img, lbls in loader:
        show_prediction(model, img, lbls)
    plt.show()

test_examples(f"{DATA_PATH}/saved_models/shader_3_predictor", f"{DATA_PATH}/example_data_3")

"""
run_cnn(f"{DATA_PATH}/train_data_1",
        f"{DATA_PATH}/test_data_1",
        save_to=f"{DATA_PATH}/saved_models/test_shader_1",
        num_epochs=50)
"""