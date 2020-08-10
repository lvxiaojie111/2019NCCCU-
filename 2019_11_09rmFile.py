##深度学习过程中，需要制作训练集和验证集、测试集。

import os, random, shutil


def moveFile(fileDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.5# 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print(sample)
    for name in sample:
        shutil.move(fileDir + name, tarDir + name)
    return
    #deleter x.jpg from file
    # pathDir=os.listdir(fileDir)
    # filenumber=len(pathDir)
    #
    # for i in range(1,5001):
    #     a=str(i)+".jpg"
    #     print(a)
    #     # for j in pathDir:
    #     #
    #     #     print(j)
    #     #     if(i==j):
    #     os.remove(os.path.join(fileDir,a))
    #     print("%s have succeed move"%os.path.join(fileDir,a))


if __name__ == '__main__':
    fileDir = "/home/ljj/share/demo3/ncccu_competition/zuoye/cnn_captcha-master/train _2/"  # 源图片文件夹路径
    tarDir = '/home/ljj/share/demo3/ncccu_competition/zuoye/cnn_captcha-master/test_2/'  # 移动到新的文件夹路径
    moveFile(fileDir)
















