import os
#read pic from col
from PIL import Image
#array col
import numpy as np
import tensorflow as tf

#data file
data_dir="data"
#train or test
train=True
#MODEL PATH
model_path="model/image_model"
#read  pic and label from file
#label form:1_400.jpg
def read_data(data_dir):
    datas=[]
    labels=[]
    fpaths=[]
    for fname in os.listdir(data_dir):
        fpath=os.path.join(data_dir,fname)
        fpaths.append(fpath)
        image=Image.open(fpath)
        data=np.array(image)/255.0
        label=int(fname.split("_"[0]))
        datas.append(data)
        labels.append(label)
    datas=np.array(datas)
    labels=np.array(labels)
    print("shape of datas:{}\tshappe of labels:{}".format(datas.shape,labels.shape))
    return fpaths,datas,labels
fpaths,datas,labels=read_data(data_dir)
#computer how class pic
num_classes=len(set(labels))

#define placeholder,put input and labels
datas_placeholder=tf.placeholder(tf.float32,[None,32,32,3])
labels_placeholder=tf.placeholder(tf.int32,[None])
#cun fang dropout can shu de rong qi,xun lian shi wei 0.25,test 0
dropout_placeholdr=tf.placeholder(tf.float32)

#define cnn :
#cnn core:20  size:5 jihuo:relu
conv0=tf.layers.conv2d(datas_placeholder,20,5,activation=tf.nn.relu)
#define max-pooling ,pooling size:2x2,step:2x2
pool0=tf.layers.max_pooling2d(conv0,[2,2],[2,2])

#define cnn :
#cnn core:40  size:4 jihuo:relu
conv1=tf.layers.conv2d(pool0,40,4,activation=tf.nn.relu)
#define max-pooling ,pooling size:2x2,step:2x2
pool1=tf.layers.max_pooling2d(conv1,[2,2],[2,2])

#conv feather dim:3 to feature dim:1
flatten=tf.layers.flatten(pool1)

#full connect,convert feature vt of 100 long
fc=tf.layers.dense(flatten,400,activation=tf.nn.relu)

#add dropout to admit  out nihe
dropout_fc=tf.layers.dropout(fc,dropout_placeholdr)

# no activate out
logits=tf.layers.dense(dropout_fc,num_classes)
predicted_labels=tf.arg_max(logits,1)
#define cross loss fun
losses=tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(labels_placeholder,num_classes),
    logits=logits
)
#define avrage loss
mean_loss=tf.reduce_mean(losses)
#define adam,appoint adam loss fun.
optimizer=tf.train.AdamOptimizer(learning_rate=1e-2.minimize(losses))

#used for ssave and resolve model
saver=tf.train.Saver()

with tf.Session() as sess:
    if train:
        print("train model")
        #if train,init para
        sess.run(tf.global_variables_initializer())
        #define input and label to cover bottle,when train,dropout is 0.25
        train_feed_dict={
            datas_placeholder:datas,
            labels_placeholder:labels,
            dropout_placeholdr:0.25
        }
        for step in range(150):
            _,mean_loss_val=sess.run([optimizer,mean_loss],feed_dict=train_feed_dict)
            if step %10==0:
                print("step={}\tmean_loss={}".format(step,mean_loss_val))
        saver.save(sess,model_path)
        print("train is over,save model to{}".format(model_path))
    else:
        print("test mode")
        #if test ,import para
        saver.restore(sess,model_path)
        print("from{}model import".format(model_path))
        #label and mingcheng dui zhao guan xi
        label_name_dict={
            0:"flying",
            1:"car",
            2:"bird"
        }
        # define input and label to cover bottle,when test,dropout is 0
        test_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            dropout_placeholdr: 0
        }
        predicted_labels_val=sess.run(predicted_labels,feed_dict=test_feed_dict)
        #real label and model predit label
        for fpath,real_label,predicted_label in zip(fpaths,labels,predicted_labels_val):
            #put label_id to label
            real_label_name=label_name_dict[real_label]
            predicted_label_name=label_name_dict[predicted_label]
            print("{}\t{}=>{}".format(fpath,real_label_name,predicted_label_name))

