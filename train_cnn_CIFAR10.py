import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2023, 0.1994, 0.2010))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=2)

    
    testset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100,
                                            shuffle=False, num_workers=0)

    print("Lengths of datasets:", len(trainset), len(testset))

    class CIFAR10_CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                # Input: 3x32x32
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 16x16

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),  # 8x8

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)  # 4x4
            )
            self.classifier = nn.ModuleDict({
                "flatten": nn.Flatten(),
                "fc1": nn.Linear(256*4*4, 512),
                "relu": nn.ReLU(inplace=True),
                "dropout": nn.Dropout(0.5),
                "fc2": nn.Linear(512, 10)
                })
                

        def forward(self, x):
            x = self.features(x)

            # For classifier we treat it as a Module Dict to extract the penultimate layer output
            for key, layer in self.classifier.items():
                x = layer(x)
                if key == "fc1":
                    penultimate_output = x  # Save the output after the first fully connected layer

            return x, penultimate_output
        

    net = CIFAR10_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    start_time = time.time()
    for epoch in range(3):  # about 20 epochs gets you ~80-85% acc
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            #print(inputs.shape)
            optimizer.zero_grad()
            outputs,_ = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {running_loss/len(trainloader):.3f}")

    print(f"Training finished. Time: {time.time() - start_time:.2f}. Num workers: {trainloader.num_workers}")


    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs,_ = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()