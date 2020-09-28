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
