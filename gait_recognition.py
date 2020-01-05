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

class Gait_rec(object):
    # 父类 用来初始化神经网络的参数以及tensorflow的基本配置
    def __init__(self,batch,epoch,train_steps,num_classes,is_train,learn_rate,trainset_dir,valset_dir):
        # 定义模型的参数
        # self.input_pose_tensor = input_pose_tensor
        self.num_classes = num_classes
        
        # 训练部分的参数
        self.is_train = is_train
        self.train_steps = train_steps
        self.epoch = epoch
        self.trainset_dir = trainset_dir
        self.valset_dir = valset_dir
        self.batch = batch
        # 定义训练的过程中可能使用到的参数以及placeholder
        if is_train:
            self.input_x = tf.placeholder(
                dtype=tf.float32,
                shape=[None,None,17,17,32],
                name="input_pic"
                )

            self.output_y = tf.placeholder(
                dtype=tf.int32,
                shape=[None,self.num_classes],                                                  
                name="persons"
                )

            self.learn_rate = tf.Variable(
                learn_rate,
                dtype=tf.float32
                )
            # 定义训练过程中的op
            # 预处理
            # self.input_x = self.tensor_preprocess(self.input_x)
            self.logits = self.get_network(self.input_x)
            self.loss_funtion = self.return_loss(self.logits)
            self.optimizer = self.return_optimizer()
            self.train_process = self.optimizer.minimize(self.loss_funtion,name="train_process")
            self.prediction = self.return_prediction()
            self.accuracy = self.return_accuracy()
            # tensorboard
            self.meraged = self.before_write_summry(self.loss_funtion,self.accuracy)
            tf.add_to_collection('loss',self.loss_funtion)
            tf.add_to_collection('acc',self.accuracy)
            tf.add_to_collection('predict',self.logits)
            # 声明sess和全局初始化
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def get_initialized_variables(self):

        global_vars = tf.global_variables()
        is_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        initialized_vars = [v for (v, f) in zip(global_vars, is_initialized) if  f]
        print([str(i.name) for i in initialized_vars])
        return initialized_vars


    def before_write_summry(self,loss,acc):
        
        loss_sum = tf.summary.scalar('loss',loss)
        acc_sum = tf.summary.scalar('acc',acc)
        meraged = tf.summary.merge([loss_sum,acc_sum])    
        return meraged
    

    def return_accuracy(self):
        acc = tf.equal(tf.argmax(self.prediction,1),tf.argmax(self.output_y,1))
        acc = tf.cast(acc,dtype=tf.float32)
        acc = tf.reduce_mean(acc,name="accuracy")
        return acc

    def return_prediction(self):
        self.prediction = tf.nn.softmax(self.logits,name="prediction")
        return self.prediction


    def return_optimizer(self):
        adam_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learn_rate,
            name="Adam"
        )
        return adam_optimizer

        
    def return_loss(self,predict_value):
        
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=predict_value,labels=self.output_y)
        mean_loss = tf.reduce_mean(loss)
        return mean_loss

   
    def residual_block(self,input_tensor,inner_hiddens,output_hiddens,stride=1):
        """
        残差模块
        """
        input_of_res = input_tensor
        if stride > 1:  
            print(stride)
            print(output_hiddens)
            # input_tensor = tf.layers.conv2d(input_tensor,output_hiddens,1,strides=(stride,stride),padding="same",kernel_initializer=tf.keras.initializers.lecun_uniform())
            input_tensor = tf.layers.conv3d(input_tensor,output_hiddens,1,strides=(stride,stride,stride),padding='same',kernel_initializer=tf.keras.initializers.lecun_uniform(),bias_initializer=tf.keras.initializers.lecun_uniform())

        in_net = tf.layers.batch_normalization(input_of_res)
        in_net = tf.nn.relu(in_net)
        # in_net = tf.layers.conv2d(in_net,inner_hiddens,1,padding="same",kernel_initializer=tf.keras.initializers.lecun_uniform())
        in_net = tf.layers.conv3d(in_net,inner_hiddens,1,padding="same",kernel_initializer=tf.keras.initializers.lecun_uniform(),bias_initializer=tf.keras.initializers.lecun_uniform())

        in_net = tf.layers.batch_normalization(in_net)
        in_net = tf.nn.relu(in_net)
        # in_net = tf.layers.conv2d(in_net,inner_hiddens,3,padding="same",strides=stride,kernel_initializer=tf.keras.initializers.lecun_uniform())
        in_net = tf.layers.conv3d(in_net,inner_hiddens,3,padding="same",strides=stride,kernel_initializer=tf.keras.initializers.lecun_uniform(),bias_initializer=tf.keras.initializers.lecun_uniform())

        in_net = tf.layers.batch_normalization(in_net)
        in_net = tf.nn.relu(in_net)
        # in_net = tf.layers.conv2d(in_net,output_hiddens,1,padding="same",kernel_initializer=tf.keras.initializers.lecun_uniform())
        in_net = tf.layers.conv3d(in_net,output_hiddens,1,padding="same",kernel_initializer=tf.keras.initializers.lecun_uniform(),bias_initializer=tf.keras.initializers.lecun_uniform())
        print(in_net)
        print(input_tensor)
        net = tf.nn.relu(in_net+input_tensor)
        return net


   
    def company_data_process(self,file_add,des):
        '''
        读取公司的数据
        file_add:训练集,验证集和测试集的读取方法
        文件路径:data/train/...
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
            son_folder_add = file_add+"/"+son_folder_name
            images = os.listdir(son_folder_add)
            number_of_frames = len(images)
            # 读取数据,并封装成pose_estimation模型可处理的形状
            # 每一个for循环处理一个子文件夹中的数据,组成一个batch
            time1 = time.time()
            a+=1
            for i in range(1,number_of_frames+1):
                img = imread(son_folder_add+"/"+str(i)+".jpg")
                img = imresize(img,[299,299])
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
        all_batch_label_numpy = tf.one_hot(all_batch_label_numpy,self.num_classes)
        sess = tf.Session()
        all_batch_label_numpy = sess.run(all_batch_label_numpy)
        # 返回
        return all_batch_image_numpy,all_batch_label_numpy


        

    
    def Human_pose_estimation(self,file_add,des):
        """
        此方法的目的是调用训练好的HumanPose模型
        将所有的图片帧处理成pose 描述向量
        """
        # 恢复模型
        # 处理训练集数据
        spatial_feature_list_train = []
        net_pose = HumanPoseIRNetwork()
        net_pose.restore("./models_pose/MPII+LSP.ckpt")
        # img,label = self.casia_data_process(self.trainset_dir)
        img ,label = self.company_data_process(file_add,des)
        # 使用该模型生成pose描述向量
        
        print("生成训练集的空间pose特征向量")
        i=0
        for image in img: 
            # print(batch_image.shape)
            print("处理第"+str(i)+"个batch,维度信息为"+str(np.array(image).shape))
            spatial_feature = net_pose.feed_forward_features(image)
            spatial_feature_list_train.append(np.array(spatial_feature))
            i+=1
        # 所有的训练集视频帧的pose特征向量提取完毕

        # 验证集的处理方式
        # spatial_feature_list_val = []
        # val_img,val_label = self.company_data_process(self.valset_dir,'val')
        # print("生成验证集数据的空间pose特征向量")
        # i_=0
        # for batch_image_ in val_img: 
        #     # print(batch_image.shape)
        #     print("处理第"+str(i_)+"个batch,维度信息为"+str(np.array(batch_image_).shape))
        #     spatial_feature_ = net_pose.feed_forward_features(batch_image_)
        #     print(spatial_feature_.shape)
        #     spatial_feature_list_val.append(np.array(spatial_feature_))
        #     i_+=1
        # 返回所有的特征供下一个模型使用

        return spatial_feature_list_train,label
            

    @abstractmethod
    def get_network(self,input_tensor):
        pass

    


    def tensor_preprocess(self,input_tensor):
        
        return input_tensor/100.0

# 子类定义网络结构,并定义训练过程
class GaitNetwork(Gait_rec):
    
    def __init__(self,batch,epoch, train_steps, num_classes, is_train, learn_rate,trainset_dir,valset_dir):
        self.epoch = epoch 
        self.train_steps = train_steps
        self.num_classes = num_classes
        self.is_train = is_train
        self.learn_rate = learn_rate
        # self.rnn_hiddens = rnn_hiddens
        # self.rnn_layers = rnn_layers
        self.valset_dir = valset_dir
        self.trainset_dir = trainset_dir

        super().__init__(batch,epoch,train_steps, num_classes, is_train, learn_rate,trainset_dir,valset_dir)


    def get_network(self, input_tensor):
        # 将输入tensor用net变量存储
        net = input_tensor
        # net = tf.layers.batch_normalization(input_tensor)
        # print("处理后的输入")
        # print(net)
        # 此方法为父类的抽象方法,主要用来构建Res_RNN结构,为训练代码提供基础
        with tf.variable_scope("GaitNN"):
            with tf.variable_scope("ResNet_process"):
                with tf.variable_scope("1717"):
                    # net = tf.layers.conv2d(net,256,1)
                    net = tf.layers.conv3d(net,256,1)
                    # 定义三个残差块
                    net = self.residual_block(net,inner_hiddens=64,output_hiddens=256)
                    net = self.residual_block(net,inner_hiddens=64,output_hiddens=256)
                    net = self.residual_block(net,inner_hiddens=64,output_hiddens=256)
                    print("res")
                    print(net)
                
                with tf.variable_scope("88"):
                    net = self.residual_block(net,inner_hiddens=64,output_hiddens=512,stride=2)
                    net = self.residual_block(net,inner_hiddens=128,output_hiddens=512)
                    net = self.residual_block(net,inner_hiddens=128,output_hiddens=512)

                with tf.variable_scope("44"):
                    net = self.residual_block(net,inner_hiddens=128,output_hiddens=512,stride=2)
                    net = self.residual_block(net,inner_hiddens=256,output_hiddens=512)

                    net = tf.layers.conv3d(net,filters=512,kernel_size=1)
                    net = tf.layers.conv3d(net,filters=512,kernel_size=3)

                with tf.variable_scope("FullyConnected"):
                # 最终网络的输出是[batch,?,5,5,256] 
                    # net = tf.layers.flatten(net)
                    net = tf.keras.layers.GlobalAveragePooling3D()(net)
                    net = tf.nn.dropout(net,0.7)
                    net = tf.layers.dense(net,self.num_classes,kernel_initializer=tf.keras.initializers.he_uniform(),bias_initializer=tf.keras.initializers.he_uniform())
                    print(net)
                    return net



    def shuffle_and_return_train_data(self,length,data,label):

        # 生成随机数
        random_index_list = [random.randint(0,length-1) for _ in range(self.batch)]
        batch_data = [np.array(data[index]) for index in random_index_list]
        batch_label = [np.array(label[index_]) for index_ in random_index_list]
        batch_data_ = np.array(batch_data)
        batch_label_ = np.reshape(np.array(batch_label),[-1,self.num_classes])
        return batch_data_,batch_label_

    def shuffle_and_return_val_data(self,data,label):
        batch_val_data = np.array(data)
        batch_val_label = np.array(label)
        batch_val_label = np.reshape(batch_val_label,[-1,self.num_classes])
        return batch_val_data,batch_val_label


    
    def train_of_model(self,save_model_file):
        # 调用父类的方法,获得所有的pose特征向量,也就是输入和对应的标签
        if not os.path.isdir(save_model_file):
            os.makedirs(save_model_file)
        print("pose estimation.......")
        train_spatial_feature , train_label = self.Human_pose_estimation(self.trainset_dir,'train')
        val_pose_feature,val_label = self.Human_pose_estimation(self.valset_dir,'val')
        print("预处理中.....")
        train_spatial_feature = self.tensor_preprocess(np.array(train_spatial_feature))
        val_feature = self.tensor_preprocess(np.array(val_pose_feature))

        # tensorboard保存训练集和验证集曲线
        writer_train = tf.summary.FileWriter("./tensorboard/train/",self.sess.graph)
        writer_val = tf.summary.FileWriter("./tensorboard/val/") 
        
        # saver 保存模型
        # saver = tf.train.Saver()
        # saver.save(self.sess,save_model_file)
        # 临时变量
        count=0
        temp_val_loss = 0

        # 训练的过程
        for e in range(self.epoch):
            print("epoch:"+str(e))
            temp_acc_of_train = 0
            for i in range(self.train_steps):
                # 生成随机数,随机选择输入的batch
                '''
                此处根据batch的参数大小,
                生成随机数,然后当做data和label的下标,随机选取数据,
                类似于tensorflow的shuffle,不过这里没有使用tensorflow.data的API
                而是手动实现的.
                '''
                # 拿到本次训练的数据
                batch_data_ , batch_label_ = self.shuffle_and_return_train_data(len(train_spatial_feature),train_spatial_feature,train_label)
                self.sess.run(self.train_process,feed_dict={self.input_x:batch_data_,self.output_y:batch_label_})
                loss,acc = self.sess.run([self.loss_funtion,self.accuracy],feed_dict={self.input_x:batch_data_,self.output_y:batch_label_})
                logits = self.sess.run(self.logits,feed_dict={self.input_x:batch_data_,self.output_y:batch_label_})
                # temp_acc_of_train+=acc
                if i%5==0:
                    print("Iteration:"+str(i))
                    print("loss: "+str(loss)+"||"+"accuracy: "+str(acc))

                # 验证模块
                # 获取验证集数据
                batch_val_data,batch_val_label = self.shuffle_and_return_val_data(val_feature,val_label)
                # epoch为0时,临时存储个loss,用于后续比较
                if e==0 and i==10:
                    val_loss,val_acc = self.sess.run([self.loss_funtion,self.accuracy],feed_dict={self.input_x:batch_val_data,self.output_y:batch_val_label})
                    # 存储val_loss的初值
                    temp_val_loss = val_loss
                if i%50==0:
                    # 每50 进行一次验证
                    val_loss,val_acc = self.sess.run([self.loss_funtion,self.accuracy],feed_dict={self.input_x:batch_val_data,self.output_y:batch_val_label})
                    print("val_loss: "+str(val_loss)+"||"+"val_acc: "+str(val_acc))
                    if (val_loss-temp_val_loss)<0:
                        # 如果当次的验证集损失比上次的低,则保存模型 即只要验证集损失降低 则保存模型
                        print("验证集损失下降"+str(temp_val_loss)+"------"+str(val_loss))
                        print("保存模型")
                        temp_val_loss = val_loss
                        tf.train.Saver(var_list=self.get_initialized_variables()).save(self.sess,save_model_file+'gait_model')
                    else:
                        print("验证集损失未下降"+str(val_loss))



                # 写tensorboard
                mer_train = self.sess.run(self.meraged,feed_dict={self.input_x:batch_data_,self.output_y:batch_label_})
                writer_train.add_summary(mer_train,count)
                mer_val = self.sess.run(self.meraged,feed_dict={self.input_x:batch_val_data,self.output_y:batch_val_label})
                writer_val.add_summary(mer_val,count)
                count+=1





    '''
    以下方法与训练公司的数据无关,是用来处理中科院开源数据集的数据预处理方法
    '''
    def casia_data_process(self,train_file):
        all_batch_img_list = []
        all_batch_label_list = []
        batch_img_list = []
        batch_label_list = []
        images_number = 0
        images_number_list = []
        
        all_sub_floder = os.listdir(train_file)
        for sub in all_sub_floder:
            sub_ = str(sub)
            # print(sub_)
            sub_floder = os.listdir(train_file+"/"+sub_) 
            
            for su in sub_floder:
                su_ =str(su)
                # print(su_)
                sub_sub_folder = os.listdir(train_file+"/"+sub_+"/"+su_)
                # 判断文件夹内的图片数量是否超过100
                images_number = len([x for x in os.listdir(train_file+str(s)+"/"+sub_+"/"+su_)])
                # images_number_list.append(images_number)
                time1 = time.time()
                for i in range(1,images_number+1):
                    if i<100:
                        # 两位数
                        if i<10:
                            img = imread(train_file+"/"+sub_+"/"+su_+"/"+sub_+"-"+su_+"-"+"00"+str(i)+".png")
                        else:
                            img = imread(train_file+"/"+sub_+"/"+su_+"/"+sub_+"-"+su_+"-"+"0"+str(i)+".png")

                    else:
                        img = imread(train_file+"/"+sub_+"/"+su_+"/"+sub_+"-"+su_+"-"+str(i)+".png")

                    img = imresize(img,[299,299])
                    batch_img_list.append(img.tolist())

                all_batch_img_list.append(batch_img_list)
                all_batch_label_list.append(self.name_to_label(str(sub_)))
                time2 = time.time()
                batch_img_list = []
                print("当前文件夹:"+sub_+"/"+su_+"下"+","+"共"+str(images_number)+"张图片处理完毕,用时"+str(time2-time1))

        all_batch_img_list = np.array(all_batch_img_list)
        all_batch_label_list = np.array(all_batch_label_list)
        all_batch_label_list = tf.one_hot(all_batch_label_list,self.num_classes)
        sess = tf.Session()
        all_batch_label_list = sess.run(all_batch_label_list)
        return all_batch_img_list,all_batch_label_list

    # 此方法不需要考虑
    def name_to_label(self,name):
        if name == "fyc":
            return 0
        elif name == "hy":
            return 1 
        elif name== "ljg":
            return 2 
        elif name== "lqf":
            return 3
        elif name=="lsl":
            return 4
        elif name=="ml":
            return 5
        elif name=="nhz":
            return 6
        elif name=="rj":
            return 7
        elif name=="syj":
            return 8
        elif name=="wl":
            return 9
        elif name=="wq":
            return 10
        elif name=="wyc":
            return 11
        elif name=="xch":
            return 12
        elif name=="xxj":
            return 13
        elif name=="yjf":
            return 14
        elif name=="zc":
            return 15
        elif name=="zdx":
            return 16
        elif name=="zjg":
            return 17
        elif name=="zl":
            return 18
        elif name=="zyf":
            return 19

