import torch
import torch.nn as nn
import torchvision


class CNN(nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.bnc1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.bnc1_2 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bnc2 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.bnc2_2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.bnc3 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.bnc3_2 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.bnc4 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bnc4_2 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1)
        self.bnc5 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.bnc5_2 = nn.BatchNorm2d(512)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Shape (batch_size, 512, 1, 1)
        self.Flatten = nn.Flatten() # Shape (batch_size, 512)

        self.fc1 = nn.Linear(512, 1024)
        self.bnf1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bnf2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, output_dim)


        self.Pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv1 - input: (batch_size, 3, 224, 224), output: (batch_size, 16, 112, 112)
        output = self.conv1(x)
        output = self.bnc1(output)
        output = self.Relu(output)

        output = self.conv1_2(output)
        output = self.bnc1_2(output)
        output = self.Relu(output)
        output = self.Pooling(output)

        # conv2 - input: (batch_size, 16, 112, 112), output: (batch_size, 32, 56, 56)
        output = self.conv2(output)
        output = self.bnc2(output)
        output = self.Relu(output)

        output = self.conv2_2(output)
        output = self.bnc2_2(output)
        output = self.Relu(output)
        output = self.Pooling(output)
        
        # conv3 - input: (batch_size, 32, 56, 56), output: (batch_size, 64, 28, 28)
        output = self.conv3(output)
        output = self.bnc3(output)
        output = self.Relu(output)

        output = self.conv3_2(output)
        output = self.bnc3_2(output)
        output = self.Relu(output)
        output = self.Pooling(output)

        # conv4 - input: (batch_size, 64, 28, 28), output: (batch_size, 128, 14, 14)
        output = self.conv4(output)
        output = self.bnc4(output)
        output = self.Relu(output)

        output = self.conv4_2(output)
        output = self.bnc4_2(output)
        output = self.Relu(output)
        output = self.Pooling(output)

        # conv5 - input: (batch_size, 128, 14, 14), output: (batch_size, 256, 7, 7)
        output = self.conv5(output)
        output = self.bnc5(output)
        output = self.Relu(output)

        output = self.conv5_2(output)
        output = self.bnc5_2(output)
        output = self.Relu(output)
        output = self.Pooling(output)

        # avgpool - input: (batch_size, 256, 7, 7), output: (batch_size, 256, 1, 1)
        output = self.avgpool(output)
        # Flatten - input: (batch_size, 256, 1, 1), output: (batch_size, 256)
        output = self.Flatten(output)

        # Fully conected 1
        output = self.fc1(output)
        output = self.bnf1(output)
        output = self.Relu(output)
        output = self.dropout(output)

        # Fully conected 2
        output = self.fc2(output)
        output = self.bnf2(output)
        output = self.Relu(output)
        output = self.dropout(output)

        # Fully conected 3
        output = self.fc3(output)

        return output



class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.model = torchvision.models.resnet50(weights='DEFAULT')
        self.model.fc = nn.Linear(self.model.fc.in_features, 120)

        for name, param in self.model.named_parameters():
            if name.startswith('fc'):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        num_features = self.model.fc.in_features
        # self.model.fc = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(num_features, 512),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(512),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes)
        # )

    def forward(self, X):
        return self.model(X)

    def unfreeze(self):
        for name, param in self.model.named_parameters():
            param.requires_grad = True


