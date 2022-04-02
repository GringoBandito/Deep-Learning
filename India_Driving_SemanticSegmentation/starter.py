from torchvision import utils
from basic_fcn import *
from dataloader import *
from utils import *
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time


# TODO: Some missing values are represented by '__'. You need to fill these up.

train_dataset = IddDataset(csv_file='train.csv')
val_dataset = IddDataset(csv_file='val.csv')
test_dataset = IddDataset(csv_file='test.csv')


train_loader = DataLoader(dataset=train_dataset, batch_size= __, num_workers= __, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= __, num_workers= __, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size= __, num_workers= __, shuffle=False)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.xavier_uniform(m.bias.data)        

epochs = 35     
criterion = nn.CorssEntropyLoss()# Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)

optimizer = optim.Adam(fcn_model.parameters(), lr=__)

use_gpu = torch.cuda.is_available()
if use_gpu:
    fcn_model = fcn_model.cuda()

        
def train():
    min_loss = np.inf
    for epoch in range(epochs):
        ts = time.time()
        for iter, (X, tar, Y) in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = X.cuda()# Move your inputs onto the gpu
                labels = Y.cuda()# Move your labels onto the gpu
                                    
            else:
                inputs, labels = X.cpu(),Y.cpu()# Unpack variables into inputs and labels

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
        
        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        loss = val(epoch)
        if loss < min_loss:
            min_loss = loss
            torch.save(fcn_model, 'best_model')

        
        fcn_model.train()
    


def val(epoch):
    ts = time.time()
    fcn_model.eval() # Don't forget to put in eval mode !
    #Complete this function - Calculate loss, accuracy and IoU for every epoch
    
    soft = Softmax(dim=1)
    
    # Make sure to include a softmax after the output from your model
    
def test():
	fcn_model.eval()
    #Complete this function - Calculate accuracy and IoU 
    # Make sure to include a softmax after the output from your model
    
if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()