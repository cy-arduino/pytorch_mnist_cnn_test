import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

from mylib import *

def main():
    #Step 1. prepare the training data
    train_data = torchvision.datasets.MNIST(
        root=MNIST_PATH,
        train=True,                                     
        transform=torchvision.transforms.ToTensor(), # normalize to 0.0~1,0
        download=True,
    )
    print(train_data)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)    # Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)

    #Step 2. prepare the test data(pick TEST_MNIST_TEST_SIZE samples)
    test_data = torchvision.datasets.MNIST(root=MNIST_PATH, train=False)
    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:TEST_MNIST_TEST_SIZE]/255.   # normalize to 0.0~1,0
    test_y = test_data.targets[:TEST_MNIST_TEST_SIZE]
    print(test_x.size())
    print(test_y.size())

    #Step 3. create nn module 
    cnn = CNN()
    print(cnn)

    #Step 4. training and testing
    optimizer = torch.optim.Adam(cnn.parameters(), lr=TRAIN_LR)   # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()                             # the target label is not one-hotted

    for epoch in range(TRAIN_EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        
            output = cnn(b_x)[0]            # cnn output
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            #test 
            if step % 50 == 0:
                test_output, last_layer = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, ' | train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

    #Step 5. save model
    torch.save(cnn.state_dict(),SAVED_MODEL_NAME)

if __name__ == '__main__':
    main()