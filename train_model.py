import numpy as np
import pandas as pd
import torch
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import imgaug.augmenters as iaa
from torch.utils.data import Dataset
import seaborn as sns


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        image, target = self.dataset[item]
        image = np.array(image)

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.dataset)

class DataSet:
    def __init__(self):
        self.transform = transforms.Compose([
                    iaa.Sequential([
                    iaa.Affine(rotate=(-15, 15)),
                    iaa.EdgeDetect(alpha=(0.25, 0.75))
                ]).augment_image,
                transforms.ToTensor()
                ])
        self.trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True)
        self.trainset1 = CustomDataset(self.trainset, transform=self.transform)
        self.valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False)
        self.valset1 = CustomDataset(self.valset, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset1, batch_size=64, shuffle=True)
        self.valloader = torch.utils.data.DataLoader(self.valset1, batch_size=64, shuffle=True)


data = DataSet()


class Model:
    output_size = 10
    def __init__(self):
        self.model = nn.Sequential(nn.Conv2d(1, 16, (3, 3), stride=(1, 1), padding=1),
                          nn.ReLU(),
                          # nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=1),
                          # nn.MaxPool2d((2, 2), stride=2),
                          # nn.ReLU(),
                          nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=1),
                          nn.MaxPool2d((2, 2), stride=2),
                          nn.ReLU(),
                          nn.Conv2d(16, 8, (3, 3), stride=(1, 1), padding=1),
                          nn.MaxPool2d((2, 2), stride=2),
                          nn.ReLU(),
                          Flatten(),
                          nn.Linear(392, 30),
                          nn.ReLU(),
                          nn.Linear(30, self.output_size),
                          nn.LogSoftmax(dim=1))


class ModelTrainer:
    images, labels = next(iter(data.trainloader))

    def __init__(self):
        self.model = Model().model

    def train(self):
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.003, momentum=0.9)
        time0 = time()
        epochs = 15
        for e in range(epochs):
            running_loss = 0
            for images, labels in data.trainloader:
                # Flatten MNIST images into a 784 long vector
                # Training pass
                optimizer.zero_grad()
                output = self.model(images)
                loss = criterion(output, labels)
                # This is where the model learns by backpropagating
                loss.backward()
                # And optimizes its weights here
                optimizer.step()
                running_loss += loss.item()
            else:
                print("Epoch {} - Training loss: {}".format(e, running_loss / len(
                    data.trainloader)))
        print("\nTraining Time (in minutes) =", (time() - time0) / 60)
        self.save_model()

    def save_model(self):
        torch.save(self.model, 'full_model_CNN.pt.pt')


class ModelTester:
    def __init__(self):
        correct_count, all_count = 0, 0
        for images, labels in data.valloader:
            for i in range(len(labels)):
                img = images[i].view(1, 1, 28, 28)
                with torch.no_grad():
                    logps = ModelTrainer().model(img)

                ps = torch.exp(logps)
                probab = list(ps.numpy()[0])
                pred_label = probab.index(max(probab))
                true_label = labels.numpy()[i]
                if (true_label == pred_label):
                    correct_count += 1
                all_count += 1
        print("Number Of Images Tested =", all_count)
        print("\nModel Accuracy =", (correct_count / all_count))

