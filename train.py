
import torch
import torchvision
import time
import os

path = os.path.dirname(os.path.realpath(__file__))

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 6, 5)
        torch.nn.init.xavier_uniform_(self.conv0.weight)
        self.r0 = torch.nn.ReLU()
        self.mp0 = torch.nn.MaxPool2d(2)
        self.conv1 = torch.nn.Conv2d(6, 16, 5)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.r1 = torch.nn.ReLU()
        self.mp1 = torch.nn.MaxPool2d(2)
        self.d0 = torch.nn.Linear(256, 120)
        torch.nn.init.xavier_normal_(self.d0.weight)
        self.r2 = torch.nn.ReLU()
        self.d1 = torch.nn.Linear(120, 84)
        torch.nn.init.xavier_normal_(self.d1.weight)
        self.r3 = torch.nn.ReLU()
        self.d2 = torch.nn.Linear(84, 10)
        torch.nn.init.xavier_normal_(self.d2.weight)
        self.out = torch.nn.LogSoftmax(1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.r0(x)
        x = self.mp0(x)
        x = self.conv1(x)
        x = self.r1(x)
        x = self.mp1(x)
        x = self.d0(x.view(x.shape[0], -1))
        x = self.r2(x)
        x = self.d1(x)
        x = self.r3(x)
        x = self.d2(x)
        x = self.out(x)
        return x

def report_error(model, data):
    correct = 0
    count = 0
    for (x, y) in data:
        correct += sum(torch.argmax(model(x),dim=1) == y)
        count += len(y)
    print('Accuracy: {:.4f}'.format(correct / count))
    
    
def train():
    epochs = 10
    num_threads = len(os.sched_getaffinity(0))
    batch_size = 16*num_threads
    train = torchvision.datasets.MNIST(path, train = True, download = True,
        transform = torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        train, batch_size = batch_size, shuffle = True,
        num_workers = num_threads, pin_memory = True)
    test = torchvision.datasets.MNIST(path, train = False, download = True,
        transform = torchvision.transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(
        test, batch_size = batch_size, shuffle = True,
        num_workers = num_threads, pin_memory = True)

    model = LeNet()
    adam = torch.optim.Adam(model.parameters(), lr = 3e-4)

    loss_fn = torch.nn.CrossEntropyLoss()
    t_start = time.time()
    for _ in range(epochs):
        for (x, y) in train_loader:
            adam.zero_grad()
            loss_fn(model(x), y).backward()
            adam.step()
    print('Took: {:.2f}'.format(time.time() - t_start))
    report_error(model, test_loader)
    
if __name__ == '__main__':
    train()
    
