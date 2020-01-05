import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.misc import imresize, imread

from human_pose_nn import HumanPoseIRNetwork

# 恢复模型方法
net_pose = HumanPoseIRNetwork()
net_pose.restore('./models_pose/MPII+LSP.ckpt')

for j in range(1,2):
    img = imread('./image/wy/rx.jpg')
    img = imresize(img, [299, 299])
    img_batch = np.expand_dims(img, 0)

    y, x, a = net_pose.estimate_joints(img_batch)
    # print("拿到的三个张量,在输出")
    # print(x)
    # print(y)
    # print(a)
    y, x, a = np.squeeze(y), np.squeeze(x), np.squeeze(a)

    joint_names = [
        'right ankle ',
        'right knee ',
        'right hip',
        'left hip',
        'left knee',
        'left ankle',
        'pelvis',
        'thorax',
        'upper neck',
        'head top',
        'right wrist',
        'right elbow',
        'right shoulder',
        'left shoulder',
        'left elbow',
        'left wrist'
    ]

    # Print probabilities of each estimation
    
    for i in range(16):
        print('%s: %.02f%%' % (joint_names[i], a[i] * 100))

    colors = ['r', 'r', 'b', 'm', 'm', 'y', 'g', 'g', 'b', 'c', 'r', 'r', 'b', 'm', 'm', 'c']
    for i in range(16):
        if i < 15 and i not in {5, 9}:
            plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color = colors[i], linewidth = 5)

    # plt.imshow(img)
    # plt.savefig('./pose_test/'+str(j)+'_pose.jpg')
    plt.savefig('./pose_test/'+str(j)+'.jpg')
    plt.close('all')
    print("第"+str(j)+"帧的pose预测完成")