#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:44:09 2019

@author: yimeng
"""

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn

import time

gpu_id = 0
device = torch.device("cuda:%d" % (gpu_id) if torch.cuda.is_available() else "cpu")


# hyper parameters
batch_size = 100
num_epochs = 10
learning_rate = 0.001


# load image
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


# create RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, 
                          nonlinearity='relu')
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, h0):
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out, hn

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, h0):
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out, hn


# create RNN
input_dim = 7     # input dimension
hidden_dim = 100  # hidden layer dimension
layer_dim = 2     # number of hidden layers
output_dim = 1    # output dimension

#net = RNNModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
net = GRUModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)


# use a Classification MSE loss and Adam
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# train and test
fig, ax = plt.subplots()
ii = 1

fig_test, ax_test = plt.subplots()
jj = 1

fig_time, ax_time = plt.subplots()
t0 = 0

for epoch in range(num_epochs):
    train_running_loss = 0.0
    test_running_loss = 0.0
    t0 = time.time()

    for train_num, (images, labels) in enumerate(trainloader):
        curr_img = images.clone()
        hn = torch.zeros([layer_dim, batch_size, hidden_dim]) 
        hn = hn.to(device)
        loss = 0
        for r in range(1, 31):
            for c in range(1, 31):
                in1 = curr_img[:, :, r-1,c-1] # top left
                in2 = curr_img[:, :, r-1,c, ] # top
                in3 = curr_img[:, :, r-1,c+1] # top right
                in4 = curr_img[:, :, r  ,c-1] # left
                inr = curr_img[:, 0, r  ,c  ]
                ing = curr_img[:, 1, r  ,c  ]
                inb = curr_img[:, 2, r  ,c  ]
                rgb = curr_img[:, :, r  ,c  ] 
                pad_tensor = (-1*torch.ones(batch_size,1)).to(device)
                
                train_r = torch.cat([in1.to(device), in2.to(device), in3.to(device), in4.to(device), pad_tensor, pad_tensor],1)
                train_g = torch.cat([in1.to(device), in2.to(device), in3.to(device), in4.to(device), inr.view(-1,1).to(device), pad_tensor],1)
                train_b = torch.cat([in1.to(device), in2.to(device), in3.to(device), in4.to(device), inr.view(-1,1).to(device), inr.view(-1,1).to(device)],1)
                                      
                # Forward propagation
                train_r = train_r.view(-1, 2, 7)
                train_g = train_g.view(-1, 2, 7)
                train_b = train_b.view(-1, 2, 7)
                output_r, hn = net(train_r, hn)
                output_g, hn = net(train_g, hn)
                output_b, hn = net(train_b, hn)

                outputs  = torch.cat([output_r, output_g, output_b], 1)                     
        
                # Calculate mse loss
                pixel_loss = criterion(outputs.squeeze(), rgb.to(device))
                loss += pixel_loss

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, train_num+1, len(trainloader), loss.item()))
        # Clear gradients
        optimizer.zero_grad()
        # Calculating gradients
        loss.backward(retain_graph=True)
        # Update parameters
        optimizer.step()
  
        # plot
        if (train_num+1) % 500 == 0:
            a = np.arange(1,num_epochs+1,1)
            t = np.array([a])
            s = loss.item()
            train_time = time.time() - t0
           
            if(ii <= t.size):
                ax.plot(ii, s, 'bo')
                ax_time.plot(ii, train_time, 'go')
                ii += 1
   
    # Save the model checkpoint
    torch.save(net.state_dict(), 'epoch'+str(epoch+1)+'.ckpt')
    
    # Save graphs
    ax.set(xlabel='epoch', ylabel='loss', title='Training Loss')
    ax.grid()
    ax_time.set(xlabel='epoch', ylabel='time (s)', title='Training Time')
    ax_time.grid()
    
    
    # Iterate through test dataset
    for test_num, (test_images, test_labels) in enumerate(testloader):
        curr_img = test_images.clone()
        ht = torch.zeros([layer_dim, batch_size, hidden_dim]) 
        ht = ht.to(device)
        loss = 0                            
        for r in range(1,30,2):
            for c in range(1,30,2):
                inn1 = curr_img[:, :, r-1,c-1] # top left
                inn2 = curr_img[:, :, r-1,c, ] # top
                inn3 = curr_img[:, :, r-1,c+1] # top right
                inn4 = curr_img[:, :, r  ,c-1] # left
                rgb_test = curr_img[:,:,r,c  ] 
                pad_tensor = (-1*torch.ones(batch_size,1)).to(device)
                
                # Forward propagation
                test_r = torch.cat([inn1.to(device), inn2.to(device), inn3.to(device), inn4.to(device), pad_tensor, pad_tensor],1)
                test_r = test_r.view(-1, 2, 7)
                out_r, ht = net(test_r, ht)  
                
                test_g = torch.cat([inn1.to(device), inn2.to(device), inn3.to(device), inn4.to(device), out_r.view(-1,1).to(device), pad_tensor],1)
                test_g = test_g.view(-1, 2, 7)
                out_g, ht = net(test_g, ht)
                
                test_b = torch.cat([inn1.to(device), inn2.to(device), inn3.to(device), inn4.to(device), out_r.view(-1,1).to(device), out_g.view(-1,1).to(device)],1)
                test_b = test_b.view(-1, 2, 7)
                out_b, ht = net(test_b, ht)

                predictions = torch.cat([out_r, out_g, out_b], 1)
                                        
                # Calculate mse loss
                pixel_loss = criterion(predictions.squeeze(), rgb_test.to(device))
                loss += pixel_loss
                                                                        
        print ('Epoch [{}/{}], Step [{}/{}], Testing Loss: {:.4f}' .format(epoch+1, num_epochs, test_num+1, len(testloader), loss.item()))
        
        if (test_num+1) % 100 == 0:
            a = np.arange(1,num_epochs+1,1)
            t = np.array([a])
            s = loss.item()
                        
            if(jj <= t.size):
                ax_test.plot(jj, s, 'ro')
                jj += 1
                                     
    ax_test.set(xlabel='epoch', ylabel='loss', title='Testing Loss')
    ax_test.grid()


torch.save(net.state_dict(), 'total_'+str(num_epochs)+'_epochs.ckpt')

fig.savefig("train_loss.png")
fig_time.savefig("train_time.png")
fig_test.savefig("test_loss.png")


