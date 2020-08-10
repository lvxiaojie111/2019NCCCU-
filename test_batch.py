# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
from PIL import Image
import random
import os
from sample import sample_conf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import backbone
import common
import imageio
class TestError(Exception):
    pass


class TestBatch(object):

    def __init__(self, img_path, char_set, model_save_dir, total):
        # 模型路径
        self.model_save_dir = model_save_dir
        # 打乱文件顺序
        self.img_path = img_path
        self.img_list = os.listdir(img_path)
        #random.seed(time.time())
        #random.shuffle(self.img_list)

        # 获得图片宽高和字符长度基本信息
        label, captcha_array = self.gen_captcha_text_image()

        captcha_shape = captcha_array.shape
        captcha_shape_len = len(captcha_shape)
        if captcha_shape_len == 3:
            image_height, image_width, channel = captcha_shape
            self.channel = channel
        elif captcha_shape_len == 2:
            image_height, image_width = captcha_shape
        else:
            raise TestError("图片转换为矩阵时出错，请检查图片格式")

        # 初始化变量
        # 图片尺寸
        self.image_height = image_height
        self.image_width = image_width
        # 验证码长度（位数）
        #self.max_captcha = len(label)
        self.max_captcha =4
        # 验证码字符类别
        self.char_set = char_set
        self.char_set_len = len(char_set)
        # 测试个数
        self.total = total

        # 相关信息打印
        print("-->图片尺寸: {} X {}".format(image_height, image_width))
        print("-->验证码长度: {}".format(self.max_captcha))
        print("-->验证码共{}类 {}".format(self.char_set_len, char_set))
        print("-->使用测试集为 {}".format(img_path))

        # tf初始化占位符
        self.X = tf.placeholder(tf.float32, [None, image_height,image_width,3 ])  # 特征向量
        self.Y = tf.placeholder(tf.float32, [None, self.max_captcha * self.char_set_len])  # 标签
        self.keep_prob = tf.placeholder(tf.float32)  # dropout值
        self.w_alpha = 0.01
        self.b_alpha = 0.1
        self.trainable=True
    def gen_captcha_text_image(self):
        """
        返回一个验证码的array形式和对应的字符串标签
        :return:tuple (str, numpy.array)
        """
        img_name = random.choice(self.img_list)
        #img_name=self.img_list
        print("123,",img_name)
        # 标签
        label = img_name.split(".")[0]
        #label = img_name
        # 文件
        img_file = os.path.join(self.img_path, img_name)
        captcha_image = Image.open(img_file)
        captcha_array = np.array(captcha_image)  # 向量化

        return label, captcha_array

    @staticmethod
    def convert2gray(img):
        """
        图片转为灰度图，如果是3通道图则计算，单通道图则直接返回
        :param img:
        :return:
        """
        if len(img.shape) > 2:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        else:
            return img

    def text2vec(self, text):
        """
        转标签为oneHot编码
        :param text: str
        :return: numpy.array
        """
        text_len = len(text)
        if text_len > self.max_captcha:
            raise ValueError('验证码最长{}个字符'.format(self.max_captcha))

        vector = np.zeros(self.max_captcha * self.char_set_len)

        for i, ch in enumerate(text):
            idx = i * self.char_set_len + self.char_set.index(ch)
            vector[idx] = 1
        return vector
    def model1(self):
        x = tf.reshape(self.X, shape=[-1, self.image_height, self.image_width, 3])#-1x60x100
        print(">>> input x: {}".format(x))
        _,_,input_data=backbone.darknet53(x,trainable=True)
        next_shape = input_data.shape[1] * input_data.shape[2] * input_data.shape[3]

        # wout = tf.get_variable('name', shape=[next_shape, self.max_captcha * self.char_set_len], dtype=tf.float32,
        #                        initializer=tf.contrib.layers.xavier_initializer())
        # bout = tf.Variable(self.b_alpha * tf.random_normal([self.max_captcha * self.char_set_len]))
        # dense = tf.reshape(input_data, [-1, wout.get_shape().as_list()[0]])
        # y_predict = tf.add(tf.matmul(dense, wout), bout,name="haha_out_y")
        wd1 = tf.get_variable(name='wd1', shape=[next_shape, 1024], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bd1 = tf.Variable(self.b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(input_data, [-1, wd1.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
        dense = tf.nn.dropout(dense, self.keep_prob)
        # #
        # # # 全连接层2-add2019119 accur:from 0.51 to 0.77
        wd2 = tf.get_variable(name='wd2', shape=[1024, 512], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bd2 = tf.Variable(self.b_alpha * tf.random_normal([512]))
        dense = tf.reshape(dense, [-1, wd2.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, wd2), bd2))
        dense = tf.nn.dropout(dense, self.keep_prob)
        #
        wout = tf.get_variable('name', shape=[512, self.max_captcha * self.char_set_len], dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
        bout = tf.Variable(self.b_alpha * tf.random_normal([self.max_captcha * self.char_set_len]))
        dense = tf.reshape(dense, [-1, wout.get_shape().as_list()[0]])
        y_predict = tf.add(tf.matmul(dense, wout), bout,name="haha_out_y")
        ############################################################
        # input_data = common.convolutional(dense1, (2, 4, 1024, 1024), self.trainable, 'conv52')
        # print(input_data.shape())
        # input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv53')
        # print(input_data.shape())
        # y_predict = common.convolutional(input_data, (1, 1, 512, self.max_captcha * self.char_set_len), self.trainable, 'conv54')
        # print(y_predict.shape())
        # next_shape = input_data.shape[1] * input_data.shape[2] * input_data.shape[3]
        # print(next_shape.shape())
        #############################################################
        # next_shape = dense1.shape[1] * dense1.shape[2] * dense1.shape[3]
        # print(next_shape)
        # #dense=tf.reshape(dense,[-1,1024])
        # # 全连接层1
        # wd1 = tf.get_variable(name='wd1', shape=[next_shape, 1024], dtype=tf.float32,
        #                       initializer=tf.contrib.layers.xavier_initializer())
        # bd1 = tf.Variable(self.b_alpha * tf.random_normal([1024]))
        # dense = tf.reshape(dense1, [-1, wd1.get_shape().as_list()[0]])
        # dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
        # dense = tf.nn.dropout(dense, self.keep_prob)
        #
        # # 全连接层2-add2019119 accur:from 0.51 to 0.77
        # wd2 = tf.get_variable(name='wd2', shape=[1024, 512], dtype=tf.float32,
        #                       initializer=tf.contrib.layers.xavier_initializer())
        # bd2 = tf.Variable(self.b_alpha * tf.random_normal([256]))
        # dense = tf.reshape(dense, [-1, wd2.get_shape().as_list()[0]])
        # dense = tf.nn.relu(tf.add(tf.matmul(dense, wd2), bd2))
        # dense = tf.nn.dropout(dense, self.keep_prob)
        # # 全连接层3
        # wout = tf.get_variable('name', shape=[512, self.max_captcha * self.char_set_len], dtype=tf.float32,
        #                        initializer=tf.contrib.layers.xavier_initializer())
        # bout = tf.Variable(self.b_alpha * tf.random_normal([self.max_captcha * self.char_set_len]))
        # y_predict = tf.add(tf.matmul(dense, wout), bout)
        return y_predict
    def model2(self):
        x = tf.reshape(self.X, shape=[-1, self.image_height, self.image_width, 3])  # -1x60x100
        print(">>> input x: {}".format(x))
        _, _, input_data = backbone.darknet53(x, trainable=True)
        # next_shape = dense1.shape[1] * dense1.shape[2] * dense1.shape[3]
        # print(next_shape)
        # dense=tf.reshape(dense,[-1,1024])
        # DBLX5层
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv52')
        input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv53')
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv54')
        input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv55')
        input_data = common.convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv56')
        # DBL1
        conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')

        next_shape = conv_lobj_branch.shape[1] * conv_lobj_branch.shape[2] * conv_lobj_branch.shape[3]

        # 全连接层1
        wd1 = tf.get_variable(name='wd1', shape=[next_shape, 1024], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bd1 = tf.Variable(self.b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv_lobj_branch, [-1, wd1.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
        dense = tf.nn.dropout(dense, self.keep_prob)

        # 全连接层2-add2019119 accur:from 0.51 to 0.77
        wd2 = tf.get_variable(name='wd2', shape=[1024, 512], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bd2 = tf.Variable(self.b_alpha * tf.random_normal([512]))
        dense = tf.reshape(dense, [-1, wd2.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, wd2), bd2))
        dense = tf.nn.dropout(dense, self.keep_prob)

        wout = tf.get_variable('name', shape=[512, self.max_captcha * self.char_set_len], dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
        bout = tf.Variable(self.b_alpha * tf.random_normal([self.max_captcha * self.char_set_len]))
        y_predict = tf.add(tf.matmul(dense, wout), bout)
        return y_predict
    def model(self):
        x = tf.reshape(self.X, shape=[-1, self.image_height, self.image_width, 3])
        print(">>> input x: {}".format(x))

        # 卷积层1
        wc1 = tf.get_variable(name='wc1', shape=[3, 3, 3, 32], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc1 = tf.Variable(self.b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'), bc1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_prob)

        # 卷积层2
        wc2 = tf.get_variable(name='wc2', shape=[3, 3, 32, 64], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc2 = tf.Variable(self.b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME'), bc2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)

        # 卷积层3
        wc3 = tf.get_variable(name='wc3', shape=[3, 3, 64, 128], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc3 = tf.Variable(self.b_alpha * tf.random_normal([128]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME'), bc3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)
        print(">>> convolution 3: ", conv3.shape)
        next_shape = conv3.shape[1]*conv3.shape[2]*conv3.shape[3]
        # 卷积层3 #add year2019month11day9
        # wc3 = tf.get_variable(name='wc3', shape=[3, 3, 64, 128], dtype=tf.float32,
        #                       initializer=tf.contrib.layers.xavier_initializer())
        # bc3 = tf.Variable(self.b_alpha * tf.random_normal([128]))
        # conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME'), bc3))
        # conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # conv3 = tf.nn.dropout(conv3, self.keep_prob)
        # print(">>> convolution 3: ", conv3.shape)
        # # 卷积层4
        # wc4 = tf.get_variable(name='wc4', shape=[1, 1, 128, 128], dtype=tf.float32,
        #                       initializer=tf.contrib.layers.xavier_initializer())
        # bc4 = tf.Variable(self.b_alpha * tf.random_normal([128]))
        # conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, wc4, strides=[1, 1, 1, 1], padding='SAME'), bc4))
        # # conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # # conv4 = tf.nn.dropout(conv4, self.keep_prob)
        # print(">>> convolution 4: ", conv4.shape)
        # next_shape = conv4.shape[1] * conv4.shape[2] * conv4.shape[3]
        # print(next_shape)

        # 全连接层1
        wd1 = tf.get_variable(name='wd1', shape=[next_shape, 1024], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bd1 = tf.Variable(self.b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, wd1.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
        dense = tf.nn.dropout(dense, self.keep_prob)
        # 全连接层2
        wd2 = tf.get_variable(name='wd2', shape=[1024, 512], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bd2 = tf.Variable(self.b_alpha * tf.random_normal([512]))
        dense = tf.reshape(dense, [-1, wd2.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, wd2), bd2))
        #dense = tf.nn.dropout(dense, self.keep_prob)
        # 全连接层3
        wout = tf.get_variable('name', shape=[512, self.max_captcha * self.char_set_len], dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
        bout = tf.Variable(self.b_alpha * tf.random_normal([self.max_captcha * self.char_set_len]))
        y_predict = tf.add(tf.matmul(dense, wout), bout)
        return y_predict

    def test_batch(self):
        #y_predict= tf.placeholder(tf.float32, [None, self.max_captcha * self.char_set_len])  # 标签
        #y_predict = self.model1()
        # x = tf.reshape(self.X, shape=[-1, self.image_height, self.image_width, 3])  # -1x60x100
        # y_predict, _ = common.bulid_resNet(x, self.max_captcha * self.char_set_len, training=True, usBN=True)  #
        #total = self.total
        #right = 0
        #self.X = tf.placeholder(tf.float32, [None, 40, 120, 3], name="x")  # 特征向量
        #self.Y = tf.placeholder(tf.float32, [None, self.max_captcha * self.char_set_len], name="y_")  # 标签
        #saver = tf.train.Saver()
        #####
        ckpt=tf.train.get_checkpoint_state(self.model_save_dir)
        print(ckpt.model_checkpoint_path+'meta')
        saver=tf.train.import_meta_graph(ckpt.model_checkpoint_path+'meta')
        graph = tf.get_default_graph()
        #tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]  # 得到当前图中所有变量的名称
        #print(tensor_name_list)
        ######
        ##############
        #aver = tf.train.import_meta_graph(model_path + '/alexnet201809101818.meta')  # 加载图结构
        #gragh = tf.get_default_graph()  # 获取当前图，为了后续训练时恢复变量
        #tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]  # 得到当前图中所有变量的名称
        ############
        print("11")
        with tf.Session() as sess:
            #init = tf.global_variables_initializer()
            #sess.run(init)
            # saver = tf.train.import_meta_graph("/home/ljj/share/demo3/ncccu_competition/zuoye/model/.meta")
            # saver.restore(sess, tf.train.latest_checkpoint("/home/ljj/share/demo3/ncccu_competition/zuoye/model/"))
            # graph = tf.get_default_graph()
            #
            # X = graph.get_tensor_by_name("x:0")
            # # keep_prob = graph.get_tensor_by_name("keep_prob:0")
            # Y = graph.get_tensor_by_name("y_:0")
            #saver = tf.train.Saver()
            #save_model=tf.train.latest_checkpoint('../model')

            ##############
            # graph = tf.get_default_graph()
            self.X=graph.get_operation_by_name('haha_out_x').outputs[0]
            print(self.X)
            y_ = graph.get_operation_by_name('y_').outputs[0]
            print(y_)
            keep_prob= graph.get_operation_by_name('keep_prob').outputs[0]
            print(keep_prob)
            #self.X = graph.get_tensor_by_name('haha_out_x')
            y_predict=tf.get_collection('pred_network')[0]
            print(y_predict)
            #sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
            saver.restore(sess, '../model/')

            #####################
            s = time.time()
            xuhao=[]
            predict_val=[]
            #liebiao=[[],[]]
            j=0
            a = np.zeros(shape=(5000, 2))
            #for i in range(total):
            # for filename in sorted(os.listdir(self.img_path),key=lambda x:int(x[:-4])):
            #     print("1112",filename,self.img_path)
            # self.img_list = sorted(os.listdir(self.img_path),key=lambda x:int(x[:-4]))
            # print("QQQQQ",self.img_list)

            # for i in self.img_list:
            #     # test_text, test_image = gen_special_num_image(i)
            #     #test_text, test_image = self.gen_captcha_text_image()  # 随机
            #     #test_text, _ = self.gen_captcha_text_image()  # 随机
            #     #test_image = self.convert2gray(test_image)
            #     #test_image = test_image.flatten() / 255
            #     img_file = os.path.join(self.img_path, i)
            #     captcha_image = Image.open(img_file)
            #     captcha_array = np.array(captcha_image)  # 向量化
            #     test_image=captcha_array
            #
            #     #test_image = np.reshape(test_image, [-1, self.image_height,self.image_width,3])
            #     #feed_dict={self.X:test_image}
            #     #
            #     #sess.run(self.Y, feed_dict=feed_dict)
            #     predict = tf.argmax(tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len]), 2)
            #     text_list = sess.run(predict, feed_dict={self.X: [test_image], self.keep_prob: 1.})
            #     predict_text = text_list[0].tolist()
            #     p_text = ""
            #     for p in predict_text:
            #         p_text += str(self.char_set[p])
            #     print("origin: {} predict: {}".format(i, p_text))
            #     xuhao.append(i)
            #     predict_val.append(p_text)
            #     #a[i][0]=test_text
            #     #a[i][1]=p_text
            #     #print( a[i][0], a[i][1])
            #     #liebiao.append(test_text,p_text)
            #     j+=1
            #     print(j)
            #     if i == p_text:
            #         right += 1
            #     else:
            #         pass
            #self.keep_prob = tf.placeholder(tf.float32)
            pic_names = [str(x) + ".jpg" for x in range(1, 5001)]
            pics = [imageio.imread(self.img_path + pic_name) for pic_name in pic_names]
            predict = tf.argmax(tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len]), 2)
            text_list = sess.run(predict, feed_dict={self.X: pics, keep_prob: 1.})
            print(np.shape(text_list))

            predict_text = text_list.tolist()

            print(np.shape(predict_text))

            for i in predict_text:
                p_text = ""
                for p in i:
                    p_text += str(self.char_set[p])
                print("origin: {} predict: {}".format(i, p_text))
                #xuhao.append(i)
                predict_val.append(p_text)
            e = time.time()
        print(predict_val)
        import pandas as pd
        ids = [str(x) + ".jpg" for x in range(1, 5001)]
        labels = predict_val
        df = pd.DataFrame([ids, labels]).T
        df.columns = ['ID', 'label']
        df.to_csv(path_or_buf="/home/ljj/share/demo3/ncccu_competition/zuoye/cnn_captcha-master/submission.csv", index=None)
        # rate = str(right*100/total) + "%"
        # print("测试结果： {}/{}".format(right, total))
        # print("{}个样本识别耗时{}秒，准确率{}".format(total, e-s, rate))
        # print(xuhao)
        # print(predict_val)
        # #############
        #aa=dict(zip(xuhao,predict_val))
        #print(aa)
        # import pandas as pd
        # a=[x for x in xuhao]
        # b=[x for x in predict_val]
        # df1=pd.DataFrame({'ID':a,'label':b})
        # #print(df1)
        # #df1=aa
        # df1.to_csv(path_or_buf="/home/ljj/share/demo3/ncccu_competition/zuoye/cnn_captcha-master/submission.csv",mode="a",encoding="utf-8",index=False,header=1)
        ##########

        #print(liebiao)


def main():
    test_image_dir = sample_conf["test_image_dir"]
    model_save_dir = sample_conf["model_save_dir"]
    char_set = sample_conf["char_set"]
    total = 5000
    tb = TestBatch(test_image_dir, char_set, model_save_dir, total)
    tb.test_batch()


if __name__ == '__main__':
    main()
