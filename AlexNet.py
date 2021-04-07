# Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton.
# "Imagenet classification with deep convolutional neural networks."
# Advances in neural information processing systems 25 (2012): 1097-1105.

# Dataset  in paper: ImageNet ILSVRC - 2010
# Input image : 256 X 256
# Input size : 227 X 227 X 3
# Architecture : 5 CNN and 3 MLP + softmax layer
# Batch : 128
# Momentum : 0.9
# Weight decay : 0.0005
# Initialize weight : zero-mean Gaussian distribution with SD of 0.01
# Initialize neuron biases : CNN2,4 and 5 and MLP with 1 else 0
# Learning rate : 0.01 -> 0.001 -> 0.0001
# Cycles : 90
# CIFAR 10
# Classes : CIFAR10 [‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’]
# Image size : 3X32X32

# This is my first implementation of a deep CNN on torch
# (Source : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.htmls )

# TODO
# Reducing over-fitting ( 1. Data augmentation -> Translation and horizontal reflection ;
#  2. PC multiplication with magnitudes proportional to eigenvalues times a random variable )

# test different batch size ( x and 2x ) for train and test set and bs =4, 128
# check out pin_memory= True and non_blocking= True
# https://stackoverflow.com/questions/63460538/proper-usage-of-pytorchs-non-blocking-true-for-data-prefetching
# tensor board

# Mean and Standard deviation preprocessing
#  dropout test time output multiplied by 0.5)
# Weight decay : 0.0005
# Initialize weight and baises
# analyze classes that are not fuctioning well ( code on pytoch page)
# Resize by rescaling such that shorter side to 256 and cropping central patch of 256 X 256 to 227 X 227


# Libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from AlexNet_functions import AlexNetwork
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Available device', device)


# parameters
batch_size = 64

# dataset
# Convert image from [0, 1] to [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((227, 227)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

torch.manual_seed(43)  # random seed generator
val_size = 5000  # 10% split
train_size = len(dataset) - val_size
trainset, valset = torch.utils.data.random_split(dataset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0, drop_last=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                          shuffle=False, num_workers=0, drop_last=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0, drop_last=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('train: ', len(trainset), 'validation: ', len(valset), 'test: ', len(testset))

# functions
def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()


def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))




# TensorBoard

# writer = SummaryWriter('runs/alexnet_cifar10_exp1')

# get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# img_grid = torchvision.utils.make_grid(images)
# imshow(img_grid)
# plt.show()
# print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
# writer.add_image('alexnet_images', img_grid)
# writer.add_graph(net, images)
# writer.close()

#Network
net = AlexNetwork()
net.to(device)
criterion = nn.CrossEntropyLoss()
print(net)


#initial evaluation
history = []
eopt = []
for batch in valloader:
    inputs, labels = batch[0].to(device), batch[1].to(device)
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    eopt.append({'val_loss': loss.detach(), 'val_acc': accuracy(outputs, labels)})

batch_losses = [x['val_loss'] for x in eopt]
epoch_loss = torch.stack(batch_losses).mean()
batch_accs = [x['val_acc'] for x in eopt]
epoch_acc = torch.stack(batch_accs).mean()
history.append({'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()})
print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(0, epoch_loss.item(), epoch_acc.item()))
print("training begin")
start_time = time.time()

proc_list = [{'epoch': 5, 'lr': 1e-1}, {'epoch': 5, 'lr': 1e-2}, {'epoch': 5, 'lr': 1e-3}, {'epoch': 5, 'lr': 1e-3}]
for proc in proc_list:
    print(proc)
    check_point= int(input(" Continue training: "))
    if check_point == 1:
        optimizer = optim.SGD(net.parameters(), lr=proc['lr'], momentum=0.9)
        for epoch in range(proc['epoch']):
            print("epoch count", epoch + 1)
            running_loss = 0.0
            i = 0  # batch count
            # training
            for batch in trainloader:
                inputs, labels = batch[0].to(device), batch[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.4f' % (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
                i +=1

            #validation
            eval_output = []
            for batch in valloader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                eval_output.append({'val_loss': loss.detach(), 'val_acc': accuracy(outputs, labels)})

            batch_losses = [x['val_loss'] for x in eval_output]
            epoch_loss = torch.stack(batch_losses).mean()
            batch_accs = [x['val_acc'] for x in eval_output]
            epoch_acc = torch.stack(batch_accs).mean()
            history.append({'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()})
            print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch + 1, epoch_loss.item(),
                                                                         epoch_acc.item()))





print('Finished Training')
end_time = time.time()
print( "total_time", (end_time - start_time)/3600)
plot_losses(history)
plot_accuracies(history)
plt.show()



#Saving model
PATH = './alexnet_cifar10_exp2.pth'
torch.save(net.state_dict(), PATH)


# loading model weights
# net.load_state_dict(torch.load(PATH))
net.eval()




#test sample images

dataiter = iter(testloader)
test_inpt = dataiter.next()


inputs, labels = test_inpt.to(device), test_inpt.to(device)
outputs = net(inputs)
# # print images
# imshow(torchvision.utils.make_grid(images))
# plt.show()
#
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))

#test entire network
correct = 0
total = 0
with torch.no_grad():
    for batch in testloader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % ( 100 * correct / total))


