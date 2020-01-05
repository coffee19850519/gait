import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import os
import random
from scipy.misc import imresize,imread
from human_pose_nn import HumanPoseIRNetwork
from abc import ABCMeta, abstractmethod
slim = tf.contrib.slim
import time
import argparse 
from video_to_pic import video_to_frames
'''
use Pre-trained pose-estimation model to generate pose vector 
and save them as csv data
'''


def company_data_process(file_add, des,num_classes):
    '''
    读取公司的数据
    file_add:训练集,验证集和测试集的读取方法
    文件路径:data/train'./image/frames_data/train'/...
    在train文件下,有若干的子文件夹,每一个子文件夹的文件名为label,文件夹内的帧为数据
    '''
    batch_image_list = []
    batch_label_list = []
    all_batch_image_list = []
    all_batch_label_list = []
    first_class_floders = []
    # 此处直接提供训练集的路径
    a = 0
    parents_folders = os.listdir(file_add)
    for son_folder in parents_folders:
        son_folder_name = str(son_folder)
        son_folder_add = file_add+son_folder_name
        images = os.listdir(son_folder_add)
        number_of_frames = len(images)
        # 读取数据,并封装成pose_estimation模型可处理的形状
        # 每一个for循环处理一个子文件夹中的数据,组成一个batch
        time1 = time.time()
        a += 1
        for i in range(1, number_of_frames+1):
            img = imread(son_folder_add+"/"+str(i)+".jpg")
            img = imresize(img, [299, 299])
            batch_image_list.append(img.tolist())
        # 循环结束,将batch-image_list 清空
        # 循环结束,存入另外一个batch_list ,目的是便于使用pose_estimation模型生成向量
        all_batch_image_list.append(batch_image_list)
        all_batch_label_list.append(int(str(son_folder_name).split('_')[0]))
        batch_image_list = []
        time2 = time.time()
        print(str(des)+"集读取完第"+str(a)+"个batch的帧数据"+"用时: "+str(time2-time1))
    # 将list转成numpy中array的形式,并返回提供给pose_estimation模型
    all_batch_image_numpy = np.array(all_batch_image_list)
    all_batch_label_numpy = np.array(all_batch_label_list)
    # 将label转成one-hot编码

    '''
    此处最后要改为参数化的形式
    '''
    all_batch_label_numpy = tf.one_hot(all_batch_label_numpy, num_classes)
    sess = tf.Session()
    all_batch_label_numpy = sess.run(all_batch_label_numpy)
    # 返回
    return all_batch_image_numpy, all_batch_label_numpy


def Human_pose_estimation(video_address, describe,num_classes):

    spatial_feature_list_train = []
    net_pose = HumanPoseIRNetwork()
    net_pose.restore("./models_pose/MPII+LSP.ckpt")
    img ,label = company_data_process(video_address,describe,num_classes)
    # 使用该模型生成pose描述向量
    print("生成训练集的空间pose特征向量")
    i=0
    for image in img: 
        # print(batch_image.shape)
        print("处理第"+str(i)+"个batch,维度信息为"+str(np.array(image).shape))
        spatial_feature = net_pose.feed_forward_features(image)
        print(np.array(spatial_feature).shape)
        # spatial-feature shape [298,17,17,32]
        print(spatial_feature)
        print(label[i])
        spatial_feature_list_train.append(np.array(spatial_feature))
        i+=1

    # print(spatial_feature_list_train)
    return spatial_feature_list_train,label


def Numpy_to_save_as_Tfrecord(data,label,describe):
    
    filename = './TFdata/'+describe+".tfrecords"
    writer = tf.python_io.TFRecordWriter(filename)    
    for i in range(len(data)):
        features = tf.train.Features(
            feature = {
            "data":tf.train.Feature(bytes_list = tf.train.BytesList(value = [data[i].astype(np.float32).tostring()])),
            "label":tf.train.Feature(bytes_list = tf.train.BytesList(value = [label[i].astype(np.int32).tostring()]))
            }  
        ) 
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)
        

def parse_function(example_proto):
    # get all the tfrecord files
    # features
    features = {"data":tf.FixedLenFeature((),tf.string),
                "label":tf.FixedLenFeature((),tf.string)
    }
    parsed_features = tf.parse_single_example(example_proto,features)
    data = tf.decode_raw(parsed_features['data'],tf.float32)
    label = tf.decode_raw(parsed_features['label'],tf.int32)
    return data , label

def load_tfrecords(tfrecords_file,epoch,batch,number_set):

    sess = tf.Session() 
    dataset = tf.data.TFRecordDataset(tfrecords_file)
    dataset = dataset.map(parse_function)

    dataset = dataset.shuffle(number_set)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat(epoch)
    iterator = dataset.make_one_shot_iterator()
    next_data = iterator.get_next()

    return next_data
    # while True:
    #     try:
    #         data,label = sess.run(next_data)
    #         print(data)
    #         print(label)
    #         return data,label
    #         # print(label)
    #     except tf.errors.OutOfRangeError:
    #         break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="process video data and generate pose-vector,save as TFrecords al last")

    # 添加参数
    parser.add_argument('-d','--data_dir',type=str,help="your video data path")
    parser.add_argument('-t','--data_type',type=str,help="your data type : train or test or val,Consistent with data_dir,if data_dir='train-set path',data_type='train'")
    parser.add_argument('-n','--num_classes',type=int,help="number of classes ,equal number of persons")
    args = parser.parse_args()

    # video to frames
    frames_dir = video_to_frames(args.data_dir,args.data_type)
    print(frames_dir)
    # save pose-vector as TFrecords
    data,label = Human_pose_estimation(frames_dir,args.data_type,args.num_classes)
    Numpy_to_save_as_Tfrecord(data,label,args.data_type)
    print("Hello bro,all video data was processed and generate into pose-vector in order to save as tfrecords")
    print("frames data was loceted in './image/frames_data/*'")
    print("TFrecords was located in './TFdata/'")
    


