# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import random
import os
from sample import sample_conf
from tensorflow.python.framework.errors_impl import NotFoundError

import cv2
import pandas as pd
# 设置以下环境变量可开启CPU识别
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class TrainError(Exception):
    pass


class TrainModel(object):
    def __init__(self, img_path, char_set, model_save_dir, verify=False):
        # 模型路径
        self.model_save_dir = model_save_dir

        # 打乱文件顺序+校验图片格式
        self.img_path = img_path
        self.img_list = os.listdir(img_path)
        # 校验格式
        if verify:
            self.confirm_image_suffix()
        # 打乱文件顺序
        random.seed(time.time())
        random.shuffle(self.img_list)

        # 获得图片宽高和字符长度基本信息
        label, captcha_array = self.gen_captcha_text_image(self.img_list[0])

        captcha_shape = captcha_array.shape
        captcha_shape_len = len(captcha_shape)
        if captcha_shape_len == 3:
            image_height, image_width, channel = captcha_shape
            self.channel = channel
        elif captcha_shape_len == 2:
            image_height, image_width = captcha_shape
        else:
            raise TrainError("图片转换为矩阵时出错，请检查图片格式")

        # 初始化变量
        # 图片尺寸
        self.image_height = image_height
        self.image_width = image_width
        # 验证码长度（位数）
        self.max_captcha = len(label)
        # 验证码字符类别
        self.char_set = char_set
        self.char_set_len = len(char_set)

        # 相关信息打印
        print("-->图片尺寸: {} X {}".format(image_height, image_width))
        print("-->验证码长度: {}".format(self.max_captcha))
        print("-->验证码共{}类 {}".format(self.char_set_len, char_set))
        print("-->使用测试集为 {}".format(img_path))

        # tf初始化占位符
        self.X = tf.placeholder(tf.float32, [None, image_height * image_width])  # 特征向量
        self.Y = tf.placeholder(tf.float32, [None, self.max_captcha * self.char_set_len])  # 标签
        self.keep_prob = tf.placeholder(tf.float32)  # dropout值随机失活（dropout）是对具有深度结构的人工神经网络进行优化的方法，在学习过程中通过将隐含层的部分权重或输出随机归零，降低节点间的相互依赖性（co-dependence ）从而实现神经网络的正则化（regularization），降低其结构风险（structural risk）
        self.w_alpha = 0.01
        self.b_alpha = 0.1

        # test model input and output
        print(">>> Start model test")
        batch_x, batch_y = self.get_batch(0, size=100)
        print(">>> input batch images shape: {}".format(batch_x.shape))
        print(">>> input batch labels shape: {}".format(batch_y.shape))
        self.label_dir=""
    def read_data(self):
        #import train dataset
        total_train_set_data=[]
        for train_set_file_name in os.listdir(self.label_dir):
            with open(os.path.join(self.label_dir,train_set_file_name),"r") as rf:
                file_data=pd.read_csv(rf)
                file_data=np.array(file_data.get_values(),dtype=np.str)
    def gen_captcha_text_image(self, img_name):
        """
        返回一个验证码的array形式和对应的字符串标签
        :return:tuple (str, numpy.array)
        """
        # 标签
        label = img_name.split("_")[0]
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

    def get_batch(self, n, size=128):
        batch_x = np.zeros([size, self.image_height,self.image_width,3])  # 初始化
        batch_y = np.zeros([size, self.max_captcha * self.char_set_len])  # 初始化

        max_batch = int(len(self.img_list) / size)
        # print(max_batch)
        if max_batch - 1 < 0:
            raise TrainError("训练集图片数量需要大于每批次训练的图片数量")
        if n > max_batch - 1:
            n = n % max_batch
        s = n * size
        e = (n + 1) * size
        this_batch = self.img_list[s:e]
        # print("{}:{}".format(s, e))
        for i, img_name in enumerate(this_batch):
            label, image_array = self.gen_captcha_text_image(img_name)
            image_array = self.convert2gray(image_array)  # 灰度化图片
            batch_x[i, :] = image_array.flatten() / 255  # flatten 转为一维
            batch_y[i, :] = self.text2vec(label)  # 生成 oneHot
        return batch_x, batch_y

    def confirm_image_suffix(self):
        # 在训练前校验所有文件格式
        print("开始校验所有图片后缀")
        for index, img_name in enumerate(self.img_list):
            print("{} image pass".format(index), end='\r')
            if not img_name.endswith(sample_conf['image_suffix']):
                raise TrainError('confirm images suffix：you request [.{}] file but get file [{}]'
                                 .format(sample_conf['image_suffix'], img_name))
        print("所有图片格式校验通过")

    def model(self):
        x = tf.reshape(self.X, shape=[-1, self.image_height, self.image_width, 1])#-1x60x100
        print(">>> input x: {}".format(x))

        # 卷积层1
        wc1 = tf.get_variable(name='wc1', shape=[3, 3, 1, 32], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc1 = tf.Variable(self.b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'), bc1))
        print(">>> convolution 1: ", conv1.shape)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_prob)
        print(">>> convolution 1: ", conv1.shape)
        # 卷积层2
        wc2 = tf.get_variable(name='wc2', shape=[3, 3, 32, 64], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc2 = tf.Variable(self.b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME'), bc2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)
        print(">>> convolution 2: ", conv2.shape)
        # 卷积层3
        wc3 = tf.get_variable(name='wc3', shape=[3, 3, 64, 128], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bc3 = tf.Variable(self.b_alpha * tf.random_normal([128]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME'), bc3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)
        print(">>> convolution 3: ", conv3.shape)
        next_shape = conv3.shape[1] * conv3.shape[2] * conv3.shape[3]

        # 全连接层1
        wd1 = tf.get_variable(name='wd1', shape=[next_shape, 1024], dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer())
        bd1 = tf.Variable(self.b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, wd1.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
        dense = tf.nn.dropout(dense, self.keep_prob)

        # 全连接层2
        wout = tf.get_variable('name', shape=[1024, self.max_captcha * self.char_set_len], dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initializer())
        bout = tf.Variable(self.b_alpha * tf.random_normal([self.max_captcha * self.char_set_len]))
        y_predict = tf.add(tf.matmul(dense, wout), bout)
        return y_predict

    def train_cnn(self):
        y_predict = self.model()
        print(">>> input batch predict shape: {}".format(y_predict.shape))
        print(">>> End model test")
        # 计算概率 损失
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predict, labels=self.Y))
        # 梯度下降
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
        # 计算准确率
        predict = tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len])  # 预测结果
        max_idx_p = tf.argmax(predict, 2)  # 预测结果
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, self.max_captcha, self.char_set_len]), 2)  # 标签
        # 计算准确率
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy_char_count = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        accuracy_image_count = tf.reduce_mean(tf.reduce_min(tf.cast(correct_pred, tf.float32), axis=1))
        tf.summary.scalar('falling_loss_summary', cost)
        tf.summary.scalar('accuracy_falling_count', accuracy_char_count)
        tf.summary.scalar('accuracy_falling_count', accuracy_image_count)
        merged_summary = tf.summary.merge_all()
        # 模型保存对象
        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            #将训练日志写入到logs文件夹下
            writer = tf.summary.FileWriter('./graph/mnist', sess.graph)  # 定义一个写入summary的目标文件，dir为写入文件地址  (交叉熵、优化器等定义)
            # 恢复模型
            if os.path.exists(self.model_save_dir):
                try:
                    saver.restore(sess, self.model_save_dir)
                # 判断捕获model文件夹中没有模型文件的错误
                except ValueError:
                    print("model文件夹为空，将创建新模型")
            else:
                pass
            step = 1
            #训练
            for i in range(10000):
                #随机选取batch个数
                batch_x, batch_y = self.get_batch(i, size=128)

                #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #run_metadata = tf.RunMetadata()

                # tf.summary.scalar('falling_loss_summary', cost)
                # tf.summary.scalar('accuracy_falling_count', accuracy_char_count)
                # tf.summary.scalar('accuracy_falling_count', accuracy_image_count)

                # 计算需要写入的日志数
                # merged_summary = tf.summary.merge_all()  # 将图像 训练过程等数据合并在一起# 计算需要写入的日志

                #运行梯度下降与损失
                _, cost_ ,summary_str= sess.run([optimizer, cost,merged_summary], feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.75})
                print("第{}次训练 >>> cost为 {} cost_ 为{}".format(step, cost, cost_))
                #print("第{}次训练 >>> optimizer为 {} predict为{} max_idx_p为{} max_idx_l为{} correct_pred为{}".format(step,optimizer, predict, max_idx_p,max_idx_l,correct_pred))

                writer.add_summary(summary_str, step)

                if step % 20== 0:
                    # 随机选取batch个数
                    batch_x_test, batch_y_test = self.get_batch(i, size=100)
                    acc_char = sess.run(accuracy_char_count, feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1.})
                    acc_image = sess.run(accuracy_image_count, feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1.})

                    #print("第{}次训练 >>> accuracy_char_count为 {} accuracy_image_count为 {} ".format(step, accuracy_char_count, accuracy_image_count))
                    print("第{}次训练 >>> 字符准确率为 {} 图片准确率为 {} >>> loss {}".format(step, acc_char, acc_image, cost_))

                    #tf.summary.scalar('loss_summary', cost_)
                    #tf.summary.scalar('accuracy_char_count', acc_image)
                    #tf.summary.scalar('accuracy_image_count', acc_char)

                    # 计算需要写入的日志数
                    #merged_summary = tf.summary.merge_all()  # 将图像 训练过程等数据合并在一起

                    #summary_str = sess.run(merged_summary, feed_dict={self.X: batch_x_test, self.Y: batch_y_test})

                    #writer.add_summary(summary_str, step)
                    # 图片准确率达到99%后保存并停止
                    if acc_image > 0.99:
                        saver.save(sess, self.model_save_dir)
                        break
                # 每训练500轮就保存一次
                if i % 50 == 0:
                    saver.save(sess, self.model_save_dir)
                step += 1
            saver.save(sess, self.model_save_dir)
            writer.close()

    def recognize_captcha(self):
        label, captcha_array = self.gen_captcha_text_image(random.choice(self.img_list))

        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(0.1, 0.9, "origin:" + label, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(captcha_array)
        # 预测图片
        image = self.convert2gray(captcha_array)
        image = image.flatten() / 255

        y_predict = self.model()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_save_dir)
            predict = tf.argmax(tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len]), 2)
            text_list = sess.run(predict, feed_dict={self.X: [image], self.keep_prob: 1.})
            predict_text = text_list[0].tolist()

        print("正确: {}  预测: {}".format(label, predict_text))
        # 显示图片和预测结果
        p_text = ""
        for p in predict_text:
            p_text += str(self.char_set[p])
        print(p_text)
        plt.text(20, 1, 'predict:{}'.format(p_text))
        plt.show()


def main():
    train_image_dir = sample_conf["train_image_dir"]
    char_set = sample_conf["char_set"]
    model_save_dir = sample_conf["model_save_dir"]
    tm = TrainModel(train_image_dir, char_set, model_save_dir, verify=False)
    tm.train_cnn()  # 开始训练模型
    # tm.recognize_captcha()  # 识别图片示例


if __name__ == '__main__':
    main()
