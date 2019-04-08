import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

from matplotlib import cm


# make reproducible
torch.manual_seed(1)    

# parameters
MNIST_PATH='./mnist/'
SAVED_MODEL_NAME='mnist_cnn.pt'

TRAIN_EPOCH = 1              
TRAIN_BATCH_SIZE = 50
TRAIN_LR = 0.001

TEST_MNIST_TEST_SIZE=1000

ENABLE_SK=True
if ENABLE_SK:
    try: 
        from sklearn.manifold import TSNE;
    except: 
        print('Please install sklearn for layer visualization')
        ENABLE_SK=False


# functions 
def plot_data(s, data, i, l, p):
    plt.imshow(data, cmap='gray')
    t = s + '[' + str(i) + ']'
    t = t + ", label=" + str(int(l))
    t = t + ", predict=" + str(p)
    plt.title(t)
    plt.show()

def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=5)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Sequential(           # input shape (1, 28, 28)
            nn.Conv2d(1, 20, 5, 1),           # output shape (20, 24, 24)
            nn.ReLU(),                        # activation
            nn.MaxPool2d(kernel_size=2),      # choose max value in 2x2 area, output shape (20, 12, 12)
        )
        self.conv2 = nn.Sequential(           # input shape (20, 12, 12)
            nn.Conv2d(20, 50, 5, 1),          # output shape (50, 8, 8)
            nn.ReLU(),                        # activation
            nn.MaxPool2d(2),                  # output shape (50, 4, 4)
        )
        
        # fully connected layer
        self.hidden1 = nn.Linear(50*4*4, 400) # 50*4*4 -> 400 
        self.hidden2 = nn.Linear(400, 100)    # 400 -> 100 
        self.out = nn.Linear(100, 10)         # 100 -> 10        
        
    def forward(self, x):        
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)             # flatten the output of conv2 -> 50*4*4
        x = self.hidden1(x)
        x = self.hidden2(x)
        output = self.out(x)
        return output, x                      # return x(output of hidden2) for visualization
        
        
        