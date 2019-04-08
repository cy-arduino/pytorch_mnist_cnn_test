import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
import numpy as np

from mylib import *

def loadImg():
    ret = []
    ret.append(np.array(PIL.ImageOps.invert(Image.open("my_input/0.bmp").convert("L").resize((28,28),Image.BILINEAR))))
    ret.append(np.array(PIL.ImageOps.invert(Image.open("my_input/1.bmp").convert("L").resize((28,28),Image.BILINEAR))))
    ret.append(np.array(PIL.ImageOps.invert(Image.open("my_input/2.bmp").convert("L").resize((28,28),Image.BILINEAR))))
    ret.append(np.array(PIL.ImageOps.invert(Image.open("my_input/3.bmp").convert("L").resize((28,28),Image.BILINEAR))))
    ret.append(np.array(PIL.ImageOps.invert(Image.open("my_input/4.bmp").convert("L").resize((28,28),Image.BILINEAR))))
    ret.append(np.array(PIL.ImageOps.invert(Image.open("my_input/5.bmp").convert("L").resize((28,28),Image.BILINEAR))))
    ret.append(np.array(PIL.ImageOps.invert(Image.open("my_input/6.bmp").convert("L").resize((28,28),Image.BILINEAR))))
    ret.append(np.array(PIL.ImageOps.invert(Image.open("my_input/7.bmp").convert("L").resize((28,28),Image.BILINEAR))))
    ret.append(np.array(PIL.ImageOps.invert(Image.open("my_input/8.bmp").convert("L").resize((28,28),Image.BILINEAR))))
    ret.append(np.array(PIL.ImageOps.invert(Image.open("my_input/9.bmp").convert("L").resize((28,28),Image.BILINEAR))))
    return ret

def main():
    
    #Step 1. prepare the test data
    '''
    test_data = torchvision.datasets.MNIST(root=MNIST_PATH, train=False)
    test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)/255.   # normalize to 0~1
    test_y = test_data.targets
    #print(test_x.size())
    #print(test_y.size())
    '''
    test_data=loadImg()
    test_x = torch.unsqueeze(torch.tensor(test_data), dim=1).type(torch.FloatTensor)/255.   # normalize to 0~1
    test_y = torch.tensor([0,1,2,3,4,5,6,7,8,9])


    #Step 2. create nn module and load weights
    cnn = CNN()
    print(cnn)   
    cnn.load_state_dict(torch.load(SAVED_MODEL_NAME))
    
    #Step 3. test
    test_output, last_layer = cnn(test_x)
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    print('test accuracy: %.2f' % accuracy)
    
    if ENABLE_SK:
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 100
        low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
        labels = test_y.numpy()[:plot_only]
        plot_with_labels(low_dim_embs, labels)
    
    #Step 4. print failed case
    for i in range(len(test_y)):
        if test_y[i] != pred_y[i]:
            plot_data("test_data", test_data[i], i, test_y[i], pred_y[i])
    
if __name__ == '__main__':
    main()