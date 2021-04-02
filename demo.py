import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import numpy as np
import os
import math
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import time
import random

# for reproducibility
seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Load DATA
data = sio.loadmat('samson_dataset.mat')
A = torch.from_numpy(data['A']) # true abundance
Y = torch.from_numpy(data['Y'])
E_VCA_init = torch.from_numpy(data['M1']).unsqueeze(2).unsqueeze(3).float()# Init endmember bundles by VCA
L = Y.shape[0]
P, N = A.shape

col=95
Y=torch.reshape(Y,(L,col,col))
A=torch.reshape(A,(P,col,col))

# Network setting
batch_size = 1
EPOCH = 500
LR = 1e-2
beta = 0.5
delta = 0.001

#define dataset
class MyTrainData(torch.utils.data.Dataset):
  def __init__(self, img, gt, transform=None):
    self.img = img.float()
    self.gt = gt.float()
    self.transform=transform

  def __getitem__(self, idx):
    return self.img,self.gt

  def __len__(self):
    return 1

class NonZeroClipper(object):
    def __call__(self, module):    
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0,1e8)

class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer('one', torch.tensor(1, dtype=torch.float))
        self.loss = nn.L1Loss(size_average=False)

    def get_target_tensor(self, input):
        target_tensor = self.one
        return target_tensor.expand_as(input)

    def __call__(self, input, gamma=1e-7):
        input = torch.sum(input, 1)
        target_tensor = self.get_target_tensor(input)
        loss = self.loss(input, target_tensor)
        return gamma*loss

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(L, 128,kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(128,momentum=0.9),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv2d(128, 64,kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(64,momentum=0.9),
            nn.ReLU(), 
            nn.Conv2d(64, P, kernel_size=(1,1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(P,momentum=0.9),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
    def forward(self,x):
        abu_est1 = self.encoder(x).clamp_(0,1)
        re_result1 = self.decoder1(abu_est1)
        abu_est2 = self.encoder(re_result1).clamp_(0,1)
        re_result2 = self.decoder2(abu_est2)
        return abu_est1, re_result1, abu_est2, re_result2

def Nuclear_norm(inputs):
    _, band, h, w = inputs.shape
    input = torch.reshape(inputs, (band, h*w))
    out = torch.norm(input, p='nuc')
    return out

class SparseKLloss(nn.Module):
    def __init__(self):
        super(SparseKLloss, self).__init__()

    def __call__(self, input,decay=1e-5):
        input = torch.sum(input, 0, keepdim=True)      
        loss = Nuclear_norm(input)
        return decay*loss

def weights_init(m):
    nn.init.kaiming_normal_(net.encoder[0].weight.data)
    nn.init.kaiming_normal_(net.encoder[4].weight.data)
    nn.init.kaiming_normal_(net.encoder[7].weight.data)


train_dataset= MyTrainData(img=Y,gt=A, transform=transforms.ToTensor())
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

net=AutoEncoder()
net.apply(weights_init)
criterionSumToOne = SumToOneLoss()
criterionSparse = SparseKLloss()

model_dict = net.state_dict()
model_dict['decoder1.0.weight'] = E_VCA_init
model_dict['decoder2.0.weight'] = E_VCA_init
net.load_state_dict(model_dict)

loss_func = nn.MSELoss(size_average=True,reduce=True,reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCH // 20, gamma=0.8)
apply_clamp_inst1 = NonZeroClipper()

time_start = time.time()
for epoch in range(EPOCH):
    for i, (x,y) in enumerate(train_loader):
        scheduler.step()

        net.train()
        abu_est1, re_result1, abu_est2, re_result2 = net(x)

        loss_sumtoone = criterionSumToOne(abu_est1) + criterionSumToOne(abu_est2)
        loss_sparse = criterionSparse(abu_est1) + criterionSparse(abu_est2)
        loss_re = beta*loss_func(re_result1,x) + (1-beta)*loss_func(x,re_result2)
        loss_abu = delta*loss_func(abu_est1,abu_est2)        

        total_loss =loss_re+loss_abu+loss_sumtoone+loss_sparse

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        net.decoder1.apply(apply_clamp_inst1)
        net.decoder2.apply(apply_clamp_inst1)

        if epoch % 10 == 0:
            print('Epoch:', epoch, '| i:', i,'| train loss: %.4f' % total_loss.data.numpy(),'| abu loss: %.4f' % loss_abu.data.numpy(),'| sumtoone loss: %.4f' % loss_sumtoone.data.numpy(),'| re loss: %.4f' % loss_re.data.numpy(),'| sparse: %.4f' % loss_sparse.data.numpy())
time_end = time.time()

net.eval()
abu_est1, re_result1, abu_est2, re_result2= net(x)
abu_est1 = abu_est1/(torch.sum(abu_est1, dim=1))
abu_est1 = torch.reshape(abu_est1.squeeze(0),(P,col,col))

for i in range(P):
    plt.subplot(2, P, i+1)
    plt.imshow(abu_est1.detach().numpy()[i,:,:])
index = [2,1,0]
A = A[index,:,:]
for i in range(P):
    plt.subplot(2, P, P+i+1)
    plt.imshow(A.detach().numpy()[i,:,:])
plt.show()
print('total computational cost:', time_end-time_start)