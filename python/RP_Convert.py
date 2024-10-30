import os
import scipy.io as sio
import numpy as np
import paddle
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.image import RecurrencePlot
import matplotlib.pyplot as plt

# 数据路径
path = 'CSI amplitude data storage path'
img_path = os.listdir(path)

# 逐个读取文件并处理数据
for line in img_path:
    data = sio.loadmat(os.path.join(path, line))
    csi_data = data['traindata']  # (90, 1344)
    img_num = csi_data.shape[0]
    count = 1
    for k in range(img_num):
        sub = csi_data[k, :].reshape(1, -1)
        transformer = PiecewiseAggregateApproximation(window_size=csi_data.shape[1] // 224)
        paa_data = transformer.transform(sub)

        rp = RecurrencePlot(dimension=1, time_delay=1)
        rp_data = rp.fit_transform(paa_data)

        # 保存图像
        imagename = f"RP plot storage path" + line.split('.')[0] + '-' + str(count) + ".png"
        plt.imsave(imagename, rp_data[0])
        count += 1
        print('successful')
