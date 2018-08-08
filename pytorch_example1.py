import torch
import numpy as np
from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#没有激活函数
class simpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
#加了非线性激活函数        
class Activation_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Activation_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True) )
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2 ,out_dim))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
       
#加了批标注化和非线性激活函数       
class Batch_Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
       
batch_size = 64
learning_rate = 1e-2
num_epoches = 20

#把各种预处理操作组合到一起，transforms.ToTensor()将图片转换成pytorch中处理的对象Tensor，在转化的过程中pytorch自动将图片标准化了，也就是说结果范围是0~1
#transforms.Normalize([0.5], [0.5])表示减去0.5,再除以0.5，将图片转化到-1~1之间
data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_dataset = datasets.MNIST(root='mnist', train=True, transform=data_tf, download=True )
test_dataset = datasets.MNIST(root='mnist', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model = simpleNet(28*28, 300, 100, 10)
# model = Activation_Net(28*28, 300, 100, 10)
model = Batch_Net(28*28, 300, 100, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
num_epoches = 5
for epoch in range(num_epoches):
    print('current epoch = %d'% epoch)
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1,28*28))
        labels = Variable(labels)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i%100 == 0:
            print('current loss = %.5f'% loss.data[0])
        

model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    img = img.view(img.size(0),-1)
    img = Variable(img, volatile=True)
    label = Variable(label, volatile=True)
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.data[0]*label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.data[0]

print('Test Loss: %.6f, Acc: %.6f'%(eval_loss/(len(test_dataset)),eval_acc.numpy()/(len(test_dataset))))
