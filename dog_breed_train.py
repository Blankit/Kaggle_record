import os
import time

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torch import nn,optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models,transforms,datasets

from sklearn.model_selection import train_test_split

'''
1. 读取数据信息
查看labels信息。id是图片名称，id.jpg保存在train文件夹下。breed是狗的类别
'''
df_train = pd.read_csv('labels.csv')
submission = pd.read_csv('sample_submission.csv')
'''
2. 类别名转成数字标签
df_train.breed.unique()获取df_train这个数据表的breed列中唯一值得个数。
将120种狗的类别对应成数字标签0~119.
'''
class_to_idx = {x:i for i,x in enumerate(df_train.breed.unique())}#类别名转成数字标签
idx_to_class = {i:x for i,x in enumerate(df_train.breed.unique())}#数字标签转成类别名，便于测试时知道输出类别
df_train['target'] =  [class_to_idx[x] for x in df_train.breed]
'''
3. 划分训练集
需要调用sklearn.model_selection的train_test_split.
将练集的一部份划分出来作为验证集的目的是挑选模型，防止模型过拟合。
'''
train,val =train_test_split(df_train,test_size=0.4, random_state=0)#将训练集的一部分划分为测试集，需要调用sklearn.model_selection的train_test_split

print(len(train),len(val))

'''
4. 构建数据集
'''
class DogsDataset(Dataset):
    '''
    df: df_train,有id,breed和新增的target信息
    root_dir:图片存放的目录
    transform: 图像处理方法
    '''
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)#数据量
    
    def __getitem__(self, idx):
        img_name = '{}.jpg'.format(self.df.iloc[idx, 0])#图片名
        fullname = os.path.join(self.root_dir, img_name)#图片路径
        image = Image.open(fullname)#PIL的Image方法
        cls = self.df.iloc[idx,2]#2是target信息
        
        if self.transform:
            image = self.transform(image)
        return [image, cls]#返回PIL对象和数字标签

'''
5. 定义图像处理方法
用到`torchvision.transforms`库

'''
normalize = transforms.Normalize( #归一化
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
ds_trans = transforms.Compose([transforms.Resize(224),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               normalize])


'''
6. 定义`dataloader`
用到`from torch.utils.data` 的`DataLoader`
'''

BATCH_SIZE = 128
data_dir = '/train/'#注意地址
train_ds = DogsDataset(train, data_dir+'train/', transform=ds_trans)#形成Dataset
val_ds = DogsDataset(val, data_dir+'train/', transform=ds_trans)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)#构建Dataloader
val_dl = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=1)
datasets = {'train':train_dl,'val':val_dl}

'''
- 查看`Dataloader`是否构建成功

for data in train_dl:
  x,y  = data
  print(x.shape,y.shape)
  print(y)
  break

'''


'''
7. 定义模型
数据集是从`Imagnet`的一个子集，可以使用在这个数据集上预训练的模型，这里选用的是`resnet18`,再次基础上微调。
`NUM_CLASS `狗的种类数，即最后预测结果的维度.
`model.fc.in_features`是 `resnet18`最后一层输入神经元的个数
用`in_fc_nums`和`NUM_CLASS`作为输入和输出神经元的个数，替换`resnet18`的全连接层
'''
model = models.resnet18(pretrained=True)
NUM_CLASS = 120#狗的种类数，即最后预测结果的维度
in_fc_nums = model.fc.in_features#resnet18最后一层输入神经元的个数
fc = nn.Linear(in_fc_nums,NUM_CLASS)
model.fc = fc
model = model.cuda()

'''
8. 定义优化器及学习率的调节方法
使用的库分别是`torch.optim`,`torch.nn`,`torch.optim.lr_scheduler`
'''
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))  # 选用AdamOptimizer
#optimizer = optim.Adam(model.fc.parameters(), lr=0.001, betas=(0.9, 0.999))  # 只优化全连接层
criterion = nn.CrossEntropyLoss()  # 定义损失函数，交叉熵
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

'''
9. 模型训练
模型训练函数有以下几个功能：
- 训练模型
- 训练一个epoch后验证，记录训练的 训练和验证集的误差及精度
- 保存在验证集上精度最高的模型
- 记录每个epoch和整个训练过程的时间
'''

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.cuda().state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            since_epoch = time.time()
            if phase == 'train': #训练阶段更新学习率
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                inputs = inputs.float().cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
#                 print(outputs.data.shape)
#                 print(preds.shape)

                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()  # item(),将torch数据转成python数据（数据只有一个元素）
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                train_epoch_loss = running_loss / len(datasets[phase])
                train_epoch_acc = running_corrects / len(datasets[phase])
            if phase == 'val':
                val_epoch_loss = running_loss / len(datasets[phase])
                val_epoch_acc = running_corrects / len(datasets[phase])

            time_elapsed_epoch = time.time() - since_epoch

            # deep copy the model
            if phase == 'val' and val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                best_model_wts = model.state_dict()
        print('{} Train Loss: {:.4f} Train Acc: {:.4f} Valdation Loss: {:.4f} Valdation Acc: {:.4f} in {:.0f}m {:.0f}s'.format(
                phase, train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc, time_elapsed_epoch // 60,time_elapsed_epoch % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)#调用训练模块，得到最优模型
