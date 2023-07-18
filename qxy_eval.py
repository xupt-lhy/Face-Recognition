from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile

from dataset import ImageDataset
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere
from matplotlib import pyplot as plt

# 图像对齐和裁剪
def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[i*n//n_folds:(i+1)*n//n_folds]
        train = list(set(base)-set(test))
        folds.append([train,test])
    return folds

def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold



parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--lfw', default='../dataset/face/lfw/lfw.zip', type=str)
#parser.add_argument('--model','-m', default='./model/sphere20a.pth', type=str)
parser.add_argument('--model','-m', default='./sphere20a_19.pth', type=str)
args = parser.parse_args()

predicts=[]
net = getattr(net_sphere,args.net)()
net.load_state_dict(torch.load(args.model))
net.cuda()
net.eval()
net.feature = True

zfile = zipfile.ZipFile(args.lfw)
# 加载landmark 每张照片包括五个特征点，五个坐标
landmark = {}
with open('data/lfw_landmark.txt') as f:
    landmark_lines = f.readlines()
for line in landmark_lines:
    l = line.replace('\n','').split('\t')
    landmark[l[0]] = [int(k) for k in l[1:]]

with open('data/pairs.txt') as f:
    pairs_lines = f.readlines()[1:]

for i in range(6000):
    p = pairs_lines[i].replace('\n','').split('\t')

    if 3==len(p):
        sameflag = 1
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[2]))
    if 4==len(p):
        sameflag = 0
        name1 = p[0]+'/'+p[0]+'_'+'{:04}.jpg'.format(int(p[1]))
        name2 = p[2]+'/'+p[2]+'_'+'{:04}.jpg'.format(int(p[3]))
    org_img1 = cv2.imdecode(np.frombuffer(zfile.read(name1),np.uint8),1)
    org_img2 = cv2.imdecode(np.frombuffer(zfile.read(name2),np.uint8),1)
    
    img1 = alignment(org_img1, landmark[name1])
    img2 = alignment(org_img2, landmark[name2])
    #img2 = alignment(cv2.imdecode(np.frombuffer(zfile.read(name2),np.uint8),1),landmark[name2])
    
    #cv2.imshow("org_img1", org_img1) 
    #cv2.imshow("org_img1", org_img2) 
    #cv2.imshow("img1", img1) 
    #cv2.imshow("img2", img2) 
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # 对输出图像使用matplotlib进行展示
    #fig_new=plt.figure()
    #img_list=[[org_img1,221],[org_img2,222],[img1,223],[img2,224]]
    #for p,q in img_list:
    #    ax=fig_new.add_subplot(q)
    #    p = p[:, :, (2, 1, 0)]
    #    ax.imshow(p)
    #plt.show()

    # cv2.flip图像翻转，1水平，0垂直，-1 水平垂直； 
    imglist = [img1,cv2.flip(img1,1),img2,cv2.flip(img2,1)]
    for i in range(len(imglist)):
        imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,112,96))
        imglist[i] = (imglist[i]-127.5)/128.0
    # 将数组给堆叠起来
    img = np.vstack(imglist)
    # numpy格式转variable格式
    img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
    output = net(img)
    #得到计算结果，f1和f2 均为512维向量。
    f = output.data
    f1,f2 = f[0],f[2]
    # 计算二者的余弦相似度，加常数是防止分母为0
    cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
    predicts.append('{}\t{}\t{}\t{}\n'.format(name1,name2,cosdistance,sameflag))

# 准确性
accuracy = []
# 阈值
thd = []
#KFold k-折交叉验证
folds = KFold(n=6000, n_folds=10, shuffle=False)
thresholds = np.arange(-1.0, 1.0, 0.005)
#predicts = np.array(map(lambda line:line.strip('\n').split(), predicts))
predicts = np.array([k.strip('\n').split() for k in predicts])

for idx, (train, test) in enumerate(folds):
    best_thresh = find_best_threshold(thresholds, predicts[train])
    accuracy.append(eval_acc(best_thresh, predicts[test]))
    thd.append(best_thresh)
print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
