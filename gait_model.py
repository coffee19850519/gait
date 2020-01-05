import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from pose_estimation import load_tfrecords
import os
import random


class Gait_model:
    
    def __init__(self,num_classes,train_steps,epoch,batch,train_tfrecord,val_tfrecord,checkpoint,learn_rate,num_trainset,num_valset):

        super().__init__()
        # 定义模型的参数
        
        self.num_classes = num_classes
        self.train_steps = train_steps
        self.epoch = epoch
        self.batch = batch
        self.train_tfrecord = train_tfrecord
        self.val_tfrecord = val_tfrecord
        self.checkpoint = checkpoint
        self.num_trainset = num_trainset
        self.num_valset = num_valset
        
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
        self.logits = self.gait_block(self.input_x)
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

    def gait_block(self,input_of_model):
        
        net = input_of_model
        
        net = tf.layers.conv3d(net,256,1)
        # 定义三个残差块
        net = self.residual_block(net,inner_hiddens=64,output_hiddens=256)
        net = self.residual_block(net,inner_hiddens=64,output_hiddens=256)
        net = self.residual_block(net,inner_hiddens=64,output_hiddens=256)
        

        net = self.residual_block(net,inner_hiddens=64,output_hiddens=512,stride=2)
        net = self.residual_block(net,inner_hiddens=128,output_hiddens=512)
        net = self.residual_block(net,inner_hiddens=128,output_hiddens=512)


        net = self.residual_block(net,inner_hiddens=128,output_hiddens=512,stride=2)
        net = self.residual_block(net,inner_hiddens=256,output_hiddens=512)

        net = tf.layers.conv3d(net,filters=512,kernel_size=1)
        net = tf.layers.conv3d(net,filters=512,kernel_size=3)

        net = tf.keras.layers.GlobalAveragePooling3D()(net)
        net = tf.nn.dropout(net,0.7)
        net = tf.layers.dense(net,self.num_classes,kernel_initializer=tf.keras.initializers.he_uniform(),bias_initializer=tf.keras.initializers.he_uniform())
        return net


    def save_model(self,checkpoint_path):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if checkpoint_path[-1]!='/':
            checkpoint_path=checkpoint_path+'/'
        saver = tf.train.Saver()
        saver.save(self.sess,checkpoint_path+"model")
        print("保存模型成功")
        
    def restore_model(self,checkpoint_path):

        model_file = checkpoint_path+"model.meta"
        model_saver = tf.train.import_meta_graph(model_file)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        model_saver.restore(self.sess,latest_checkpoint)
        # 取得acc和loss
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('acc')[0]

        return loss,accuracy
    



    def train_and_val_model(self):
        '''
        模型的训练和验证
        '''
        temp_val_loss = 0
        # 读数据,读取数据的方法定义在pose_estimation方法中
        next_train_data = load_tfrecords(self.train_tfrecord,self.epoch*self.train_steps,self.batch,self.num_trainset)
        next_val_data = load_tfrecords(self.val_tfrecord,1,self.num_valset,self.num_valset)
        batch_val_data,batch_val_label = self.sess.run(next_val_data)
        batch_val_data = np.reshape(batch_val_data,[self.num_valset,-1,17,17,32])
        # print(self.epoch)
        # print(self.train_steps)
        for e in range(self.epoch):
            for i in range(self.train_steps):
                try:
                    train_data,train_label = self.sess.run(next_train_data) 
                    train_data = np.reshape(train_data,[self.batch,-1,17,17,32])
                    print(train_label)
                    self.sess.run(self.train_process,feed_dict={self.input_x:train_data,self.output_y:train_label})
                    loss , acc = self.sess.run([self.loss_funtion,self.accuracy],feed_dict={self.input_x:train_data,self.output_y:train_label})
                
                except tf.errors.OutOfRangeError:
                    break

                # if i%5==0:
                print("Iteration:"+str(i))
                print("loss: "+str(loss)+"||"+"accuracy: "+str(acc))
            
                # print(batch_val_label)
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
                        # 保存模型
                        self.save_model(self.checkpoint)
                    else:
                        print("验证集损失未下降"+str(val_loss))







# if __name__ == "__main__":
    
#     gait = Gait_model(num_classes=2,train_steps=50,epoch=10,batch=3,\
#                     train_tfrecord='./TFdata/train.tfrecords',val_tfrecord='./TFdata/val.tfrecords',\
#                     test_tfrecord='1',checkpoint='./Our_model/')


#     gait.train_and_val_model()
