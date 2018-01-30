# kNN_HandwrittenNumberRecognition
基于mnist手写数据集，用k-NN算法实现手写数字的识别
    
    python版本为3.6

    MNIST 数据集来自美国国家标准与技术研究所, National Institute of Standards and Technology (NIST).训练集(training set)
由来自 250 个不同人手写的数字构成, 其中 50% 是高中学生, 50% 来自人口普查局 (the Census Bureau)的工作人员. 测试集(test set)
也是同样比例的手写数字数据。

    项目中kNN文件夹为简单的实现k-NN算法，训练集和测试集皆为minist数据集。
    项目中mnist1加入了图片处理，手写数字可用画图软件模拟，起初添加了处理大小为32×32的图片方法imgtovector，原图片存放至
img-number目录下，将图片转换为0,1矩阵后，存放在vectorImg目录下，然后调用k近邻算法，获得识别结果；随后添加了将任意大小的
图片处理为指定大小的方法OperateImg，待测图片存放至test目录下，同样的方式，可得出识别结果。
