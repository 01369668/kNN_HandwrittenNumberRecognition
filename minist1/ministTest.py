from minist1 import minist
import minist1.imgtovector
import minist1.OperateImg as OP
import os

#1.获取处理后的图片（传入图片的大小不一，统一处理为32乘32的0,1矩阵）
testFiles = os.listdir(r"./test/")
testPic = OP.GetTestPicture(testFiles)

#2.处理32×32的图片为0,1矩阵
files = os.listdir(r"./img-number/")
for i, item in enumerate(files):
    # print(item[:-4])
    impath = "E:\Python\kNN_HandwrittenNumberRecognition\minist1\img-number\\" + item
    savepath ='E:\Python\kNN_HandwrittenNumberRecognition\minist1\\vectorImg\\' + item[:-4] + ".txt"
    minist1.imgtovector.imgtovector(impath, savepath)


minist.testHandWritingClass()


