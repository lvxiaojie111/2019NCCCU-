# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
from PIL import Image
import random
import os
from sample import sample_conf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import backbone
import common
class TestError(Exception):
    pass


class TestBatch(object):

    def __init__(self,img_path,model_save_dir,char_set):
        # 模型路径

 # 初始化变量
        # 图片尺寸
        self.image_height = 40
        self.image_width = 120
        # 验证码长度（位数）
        #self.max_captcha = len(label)
        self.max_captcha =4
        self.model_save_dir=model_save_dir
        self.img_path=img_path
        # 验证码字符类别
        self.char_set = char_set
        self.char_set_len = len(char_set)
        # 测试个数
        # 相关信息打
        # tf初始化占位符
        #self.X = tf.placeholder(tf.float32, [None,  self.image_height,self.image_width,3 ])  # 特征向量
        #self.Y = tf.placeholder(tf.float32, [None, self.max_captcha * self.char_set_len])  # 标签
        self.keep_prob = tf.placeholder(tf.float32)  # dropout值
    def test_batch(self):
        total =5000
        right = 0
        #save_model=tf.train.latest_checkpoint('../model')graph=tf.Graph()
        with tf.Session() as sess:
            graph=tf.get_default_graph()
            #tf.saved_model.loader.load(sess,"/home/ljj/share/demo3/ncccu_competition/zuoye/model/")

            print("22")
            saver = tf.train.import_meta_graph(
                "/home/ljj/share/demo3/ncccu_competition/zuoye/cnn_captcha-master/model/.meta")
            print("21")
            #saver.restore(sess, tf.train.latest_checkpoint(self.model_save_dir))
            x=graph.get_tensor_by_name('haha_out_x:0')
            y_predict=graph.get_tensor_by_name('haha_out_y:0')
            # saver = tf.train.Saver()
            # saver.restore(sess, '../model/.')
            # saver = tf.train.import_meta_graph("/home/ljj/share/demo3/ncccu_competition/zuoye/model/.meta")
            # saver.restore(sess, tf.train.latest_checkpoint("/home/ljj/share/demo3/ncccu_competition/zuoye/model/"))
            # graph = tf.get_default_graph()
            #
            # X = graph.get_tensor_by_name("x:0")
            # # keep_prob = graph.get_tensor_by_name("keep_prob:0")
            # Y = graph.get_tensor_by_name("y_:0")

            print("11")

            s = time.time()
            xuhao=[]
            predict_val=[]
            #liebiao=[[],[]]
            j=0
            a = np.zeros(shape=(5000, 2))
            #for i in range(total):
            for filename in sorted(os.listdir(self.img_path),key=lambda x:int(x[:-4])):
                print("1112",filename,self.img_path)
            self.img_list = sorted(os.listdir(self.img_path),key=lambda x:int(x[:-4]))
            print("QQQQQ",self.img_list)
            save_model=tf.train.latest_checkpoint(self.model_save_dir)
            print("222344",save_model)
            saver.restore(sess, save_model)
            for i in self.img_list:
                # test_text, test_image = gen_special_num_image(i)
                #test_text, test_image = self.gen_captcha_text_image()  # 随机
                #test_text, _ = self.gen_captcha_text_image()  # 随机
                #test_image = self.convert2gray(test_image)
                #test_image = test_image.flatten() / 255
                img_file = os.path.join(self.img_path, i)
                captcha_image = Image.open(img_file)
                captcha_array = np.array(captcha_image)  # 向量化
                test_image=captcha_array

                #test_image = np.reshape(test_image, [-1, self.image_height,self.image_width,3])
                #feed_dict={self.X:test_image}
                #
                #sess.run(self.Y, feed_dict=feed_dict)
                predict = tf.argmax(tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len]), 2)
                text_list = sess.run(predict, feed_dict={x: [test_image], self.keep_prob: 1.})
                predict_text = text_list[0].tolist()
                p_text = ""
                for p in predict_text:
                    p_text += str(self.char_set[p])
                print("origin: {} predict: {}".format(i, p_text))
                xuhao.append(i)
                predict_val.append(p_text)
                #a[i][0]=test_text
                #a[i][1]=p_text
                #print( a[i][0], a[i][1])
                #liebiao.append(test_text,p_text)
                j+=1
                print(j)
                if i == p_text:
                    right += 1
                else:
                    pass
            e = time.time()

        rate = str(right*100/total) + "%"
        print("测试结果： {}/{}".format(right, total))
        print("{}个样本识别耗时{}秒，准确率{}".format(total, e-s, rate))
        print(xuhao)
        print(predict_val)
        #############
        #aa=dict(zip(xuhao,predict_val))
        #print(aa)
        import pandas as pd
        a=[x for x in xuhao]
        b=[x for x in predict_val]
        df1=pd.DataFrame({'ID':a,'label':b})
        #print(df1)
        #df1=aa
        df1.to_csv(path_or_buf="/home/ljj/share/demo3/ncccu_competition/zuoye/cnn_captcha-master/test.csv",mode="a",encoding="utf-8",index=False,header=1)
        ##########

        #print(liebiao)


def main():
    test_image_dir = sample_conf["test_image_dir"]
    model_save_dir = sample_conf["model_save_dir"]
    char_set = sample_conf["char_set"]
    total = 5000
    tb = TestBatch(test_image_dir,model_save_dir,char_set)
    tb.test_batch()


if __name__ == '__main__':
    main()
