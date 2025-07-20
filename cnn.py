import torch
import torch.nn as nn   # Building blocks for neural networks
import torch.nn.functional as F # Various functions for building neural networks
import torch.optim as optim  # Optimiser for neural networks
from torchsummary import summary  # Summarise PyTorch model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter # Tensorboard in PyTorch
import datetime


from tqdm import tqdm # Progress bar during training


import os
import pandas as pd
import numpy as np
import graphics.graphics_utils as mn

import matplotlib.pyplot as plt

DATA_PATH = "./"

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
        #label = self.img_labels.iloc[idx, 1].reshape(1)
        label = self.img_labels.iloc[idx, 1:5]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def get_all_labels(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        _, _, x, y, z, r, _ = mn.load_screen(img_path)
        return [x, y, z, r]
    
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
    plt.figure()
    plt.title(title)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

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


def get_data_loader(img_dir, batch_size=64):
    annotations_file_path = f"{img_dir}_annotations"
    if not os.path.exists(annotations_file_path):
        make_dataset_annotations_file(img_dir, annotations_file_path)
    #transform = lambda img : transforms.Normalize((0.5,), (0.5,)).forward(torch.flip(img, [0]).to(torch.float))
    #transform = transforms.Normalize((197.6725,), (39.3379,))
    #transform = lambda img : transforms.Normalize((197.6725,), (39.3379,)).forward(img.to(torch.float))
    transform = lambda img : nn.AvgPool2d(kernel_size=4, stride=4)(transforms.Normalize((197.6725,), (39.3379,))(img.to(torch.float)))
    #transform = lambda img : transforms.Normalize((197.6725,), (256.0,)).forward(img.to(torch.float))
    #transform = lambda img : img.to(torch.float)
    target_transform = lambda lbl : torch.tensor(lbl).to(torch.float) / 400.0
    #target_transform = lambda lbl : (torch.tensor(lbl).to(torch.float) - 300.0) / 400.0
    target_transform = lambda lbl : (torch.tensor(lbl).to(torch.float) - 50.0) / 50.0
    target_transform = scale_labels
    dataset = CustomImageDataset(annotations_file_path, img_dir, transform, target_transform)
    return DataLoader(dataset, batch_size=batch_size)




"""def test_custom_dataset():
    img_dir = f"{DATA_PATH}/test_dataset"
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
    print(dataset[0])"""


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
        #self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(1800, 128)
        #self.fc2 = nn.Linear(128, 1)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        #x = self.avgpool(x)
        x = torch.relu(self.conv1(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def test_example(model, test_data_dir):
    test_dataloader = get_data_loader(test_data_dir, batch_size=1)
    img, lbl = next(iter(test_dataloader))
    output = model(img)
    print(f"Predicted value: {output}")
    print(f"True value: {lbl}")
    print(f"(error = {output - lbl})")

def show_prediction(model, img, lbls):
    pred_lbls = list(map(lambda x: round(float(x), 4), model(img)[0]))
    true_lbls = list(map(lambda x: round(float(x), 4), lbls[0]))
    title = f"True: {true_lbls}\nPredicted: {pred_lbls}"
    show_my_tensor(img, title)



def run_cnn(num_epochs=5, save_to=""):
    train_loader = get_data_loader(f"{DATA_PATH}/train_data")
    test_loader = get_data_loader(f"{DATA_PATH}/test_data")

    # Set up TensorBoard writer
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/CNN"
    writer = SummaryWriter(log_dir)

    # Instantiate the model (using cuda if it is available)
    model = CNN().cuda() if torch.cuda.is_available() else CNN()

    # Define optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    #loss_fn = nn.MSELoss(reduction='none')
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

            #print(outputs.shape, labels.shape)
            #print(type(outputs), type(labels))
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total += labels.size(0)

            running_loss += loss.item()
            all_my_running_losses.append(running_loss)

        #print(all_my_running_losses)
        #print("Total loss:", running_loss)

        #print(running_loss / len(train_loader))
        
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
        torch.save(model.state_dict(), save_to)

    # Summarise the model
    summary(model, input_size=(1, 64, 64))



def test_test():
    test_loader = get_data_loader(f"{DATA_PATH}/test_data")
    img, lbl = next(iter(test_loader))
    show_my_tensor(img[0], lbl[0])


def load_cnn(file_path):
    model = CNN().cuda() if torch.cuda.is_available() else CNN()
    model.load_state_dict(torch.load(file_path))
    return model


def main():
    model = load_cnn(f"{DATA_PATH}/saved_models/test_shader_3")
    loader = get_data_loader(f"{DATA_PATH}/example_data_3", batch_size=1)
    img, lbls = next(iter(loader))
    show_prediction(model, img, lbls)

#main()
#run_cnn(num_epochs=50, save_to=f"{DATA_PATH}/saved_models/test_shader_3")