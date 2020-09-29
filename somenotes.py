# import numpy as np
#
# # 判断输入数据是否存在nan
# if np.any(np.isnan(input.cpu().numpy())):
#     print('Input data has NaN!')
#
# # 判断损失是否为nan
# if np.isnan(loss.item()):
#     print('Loss value is NaN!')


# 当pytorch模型写明是eval()时有时表现的结果相对于train(True)差别非常巨大，这种差别经过逐层查看，主要来源于使用了BN，
# 在eval下，使用的BN是一个固定的running rate，而在train下这个running rate会根据输入发生改变。解决方案是冻住BN
# def freeze_bn(m):
#     if isinstance(m, nn.BatchNorm2d):
#         m.eval()
# net.apply(freeze_bn)
# file_list = os.listdir('D:/DIGISKY/data7')
# class MyDataset(data.Dataset):
#     def __init__(self,datatxt,transform=None,target_transform=None):
#         super(MyDataset,self).__init__()
#          fh=open(datatxt,'r')#读取标签文件.txt
#          imgs=[]#暂时定义一个空的列表
#          for line in fh:
#             line.strip('\n')#出去字符串末尾的空格、制表符
#              words=line.split()#将路径名与标签分离出来
#             imgs.append((words[0],int(words[1])))#word[0]表示图片的路径名，word[1]表示该数字图片对应的标签
#         self.imgs=imgs
#         self.transform=transform
#         self.target_transform=target_transform
#         #self.loader=loader
#     def __getitem__(self, index):
#         fn,label=self.imgs[index]#fn表示图片的路径
#         img = Image.open(fn)#.convert('RGB'),这里时候需要转换成RGB图像视神经网络结构而定，读取文件的路径名，也即打开图片
#          if self.transform is not None:
#             img=self.transform(img)
#          return img,label#返回图片与标签
#     def __len__(self):
#          return len(self.imgs)
