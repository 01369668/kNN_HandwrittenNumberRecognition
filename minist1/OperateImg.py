import os
import numpy as np
from skimage import io
from PIL import Image
from skimage import color

N = 32
color1 = 150/255
STR = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")

def JudgeEdge(img, length, flag, size):
    '''Judge the Edge of Picture判断图片切割的边界'''
    # print("imgsize",type(img))
    for i in range(length):
        #Row or Column 判断是行是列
        if flag == 0:
            #Positive sequence 正序判断该行是否有手写数字
            line1 = img[i, img[i,:]<color1]
            #Negative sequence 倒序判断该行是否有手写数字
            line2 = img[length-1-i, img[length-1-i,:]<color1]
        else:
            line1 = img[img[:,i]<color1, i]
            line2 = img[img[:,length-1-i]<color1,length-1-i]
        #If edge, recode serial number 若有手写数字，即到达边界，记录下行
        if len(line1)>=1 and size[0]==-1:
            size[0] = i
        if len(line2)>=1 and size[1]==-1:
            size[1] = length-1-i
        #If get the both of edge, break 若上下边界都得到，则跳出
        if size[0]!=-1 and size[1]!=-1:
            break
    print('图片边界值',size)
    return size

def StretchPicture(img):
    '''Stretch the Picture拉伸图像'''
    newImg1 = np.zeros(N*len(img)).reshape(len(img), N)
    newImg2 = np.zeros(N**2).reshape(N, N)
    #对每一行进行拉伸/压缩
    #每一行拉伸/压缩的步长
    temp1 = len(img[0,:])/N
    #每一列拉伸/压缩的步长
    temp2 = len(img)/N
    #对每一行进行操作
    for i in range(len(img)):
        for j in range(N):
            newImg1[i, j] = img[i, int(np.floor(j*temp1))]
    #对每一列进行操作
    for i in range(N):
        for j in range(N):
            newImg2[i, j] = newImg1[int(np.floor(i*temp2)), j]
    return newImg2

def CutPictureSize(img):
    '''Cut the Picture 切割图象'''
    #初始化新大小
    size = []
    #图片的行数
    length = len(img)
    #图片的列数
    width = len(img[0,:])
    #计算新大小
    size.append(JudgeEdge(img, length, 0, [-1, -1]))
    size.append(JudgeEdge(img, width, 1, [-1, -1]))
    size = np.array(size).reshape(4)
    return img[size[0]:size[1]+1, size[2]:size[3]+1]

def GetTestPicture(files):
    '''得到待检测图片并保存'''


    for i, item in enumerate(files):
        img = io.imread('./test/'+item)
        img = color.rgb2grey(img)
        img[img>color1] = 1
        img = CutPictureSize(img)
        img = StretchPicture(img).reshape(N,N)

        # np.savetxt()
        # img = np.loadtxt(item[0:-4]+'.txt')
        for i in range(len(img)):
            for j in range(len(img[0,:])):
                if img[i][j] == 1:
                    img[i][j] = 0
                else:
                    img[i][j] = 1
        np.savetxt('./vectorImg/' + (item[0:-4] + '.txt'), img, fmt="%d",delimiter='')
        # print(img)
        image = Image.fromarray(img)
        # print("ccc",image)