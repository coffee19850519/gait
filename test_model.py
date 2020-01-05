import tensorflow as tf 
import numpy as np
# from gait_recognition import Gait_rec,GaitNetwork
from pose_estimation import load_tfrecords
import argparse



def read_model(model_filename,test_tfrecords,num_test):
    with tf.Session() as sess:

        model_file = model_filename+"model.meta"
        model_saver = tf.train.import_meta_graph(model_file)
        checkpoint = tf.train.latest_checkpoint(model_filename)
        model_saver.restore(sess,checkpoint)

        # 拿到损失和准确率
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('acc')[0]  
        prediction = tf.get_collection('predict')[0]

        # 加载placeholder
        graph = tf.get_default_graph()

        X = graph.get_operation_by_name('input_pic').outputs[0]
        Y = graph.get_operation_by_name('persons').outputs[0]

        # 读取测试集数据
        

        next_data = load_tfrecords(test_tfrecords,1,num_test,num_test)
        test_data , test_label = sess.run(next_data)
        print(test_label)
        test_data = np.reshape(test_data,[num_test,-1,17,17,32])
        # print("共测试"+str(len(data))+"组数据")
        print("测试结果为........")
        acc_of_test = sess.run(accuracy,feed_dict={X:test_data,Y:test_label})
        print(str(acc_of_test)+"%")


if  __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test model to get acc and loss")
    # 添加参数
    parser.add_argument('-m','--model_file',type=str,help="model path")
    parser.add_argument('-t','--test_tfrecords',type=str,help="test data path ")
    parser.add_argument('-n','--num_test',type=int,help="number of test samples")
    args = parser.parse_args()
 

read_model(args.model_file,args.test_tfrecords,args.num_test)

