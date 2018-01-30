# from Tkinter import *
from PIL import  Image
from numpy import *

def imgtovector(impath,savepath):
    '''
    convert the image to an numpy array
    Black pixel set to 1,white pixel set to 0
    '''
    im = Image.open(impath)
    im = im.transpose(Image.ROTATE_90)
    im = im.transpose(Image.FLIP_TOP_BOTTOM)

    rows = im.size[0]
    cols = im.size[1]
    imBinary = zeros((rows,cols))
    for row in range(0,rows):
        for col in range(0,cols):
            imPixel = im.getpixel((row,col))[0:3]
            if imPixel == (0,0,0):
                imBinary[row,col] = 1
            else:
                imBinary[row,col] = 0
    #save temp txt like 1_5.txt whiich represent the class is 1 and the index is 5
    fp = open(savepath,'w')
    for x in range(0,imBinary.shape[0]):
        for y in range(0,imBinary.shape[1]):
            fp.write(str(int(imBinary[x,y])))
        fp.write('\n')
    fp.close()