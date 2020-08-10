from easydict import EasyDict
import os
import json
sample_conf = EasyDict()  #以让你像访问属性一样访问dict里的变量

# 图片文件夹
sample_conf.origin_image_dir = "./sample/origin/"
sample_conf.train_image_dir = "/home/ljj/share/demo3/ncccu_competition/zuoye/cnn_captcha-master/train _2"
sample_conf.test_image_dir = "/home/ljj/share/demo3/ncccu_competition/zuoye/cnn_captcha-master/test/"
sample_conf.test1_image_dir = "/home/ljj/share/demo3/ncccu_competition/zuoye/cnn_captcha-master/test_2"
sample_conf.api_image_dir = "./sample/api/"
sample_conf.online_image_dir = "./sample/online/"
sample_conf.local_image_dir = "./sample/local/"

# 模型文件夹
sample_conf.model_save_dir = "../model/"

# 图片相关参数
sample_conf.image_width = 120
sample_conf.image_height = 40
sample_conf.max_captcha =4
sample_conf.image_suffix = "jpg"

# 验证码字符相关参数
#sample_conf.char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sample_conf.char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                       'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U',
                        'V','W','X','Y','Z']
# char_set = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
# char_set = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# 如果有json文件，则读取json中的文件，并将json中的信息作为上述 验证码字符相关参数
use_labels_json_file = False
if use_labels_json_file:
    if os.path.exists("gen_image/labels.json"):
        with open("gen_image/labels.json", "r") as f:
            content = f.read()
            if content:
                sample_conf.char_set = json.loads(content)
            else:
                pass
    else:
        pass

sample_conf.remote_url = "https://www.xxxxx.com/getImg"
