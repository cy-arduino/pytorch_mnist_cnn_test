# pytorch_mnist_cnn_test

reference to following examples:
1. https://github.com/pytorch/examples/blob/master/mnist/main.py
2. https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/401_CNN.py

model:
	CNN(
	  (conv1): Sequential(
		(0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
		(1): ReLU()
		(2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
	  )
	  (conv2): Sequential(
		(0): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))
		(1): ReLU()
		(2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
	  )
	  (hidden1): Linear(in_features=800, out_features=400, bias=True)
	  (hidden2): Linear(in_features=400, out_features=100, bias=True)
	  (out): Linear(in_features=100, out_features=10, bias=True)
	)
