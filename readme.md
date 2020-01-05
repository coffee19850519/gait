<!--
 * @Author: Wang ye
 * @Date: 2020-01-05 09:09:45
 * @LastEditTime : 2020-01-05 10:56:14
 * @LastEditors  : Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: \mypapers\Access-Template\readme.md
 -->
# How to use gait model 

## File Introduction
* Folder:
  * image —— It stores the frame data after the video has been converted to a frame
  * models —— Pose-estimation pre-trained model ,if you want know more about it ,please click [here](https://github.com/marian-margeta/gait-recognition)
  * models_pose —— another Pose-estimation pre-trained model,but trained data is different from 'models'

* File
  * video_to_pic.py —— convert videos data into frames data 
  * pose_estimation.py —— estimate the pose of people of frames data,and generate the pose-vector ,save pose-vector data as TFrecords files.
  * gait_model.py —— define the gait model:3D Resnet model , able to train  and validate
  * test_model.py —— test the model use test data
  * other files

## How to use 
  1. run `python pose_estimation.py -h` to get the parameters information ,such as :

            usage: pose_estimation.py [-h] [-d DATA_DIR] [-t DATA_TYPE] [-n NUM_CLASSES]
            process video data and generate pose-vector,save as TFrecords al last
            optional arguments:
            -h, --help            show this help message and exit
            -d DATA_DIR, --data_dir DATA_DIR
                                your video data path
            -t DATA_TYPE, --data_type DATA_TYPE
                                your data type : train or test or val,Consistent with
                                data_dir,if data_dir='train-set
                                path',data_type='train'
            -n NUM_CLASSES, --num_classes NUM_CLASSES

            Therefore,when you want to run this file ,you should set these parameters. Here is the correct example
    
`CUDA_VISIBLE_DEVICES=0 python pose_estimation.py -d '/home/a524wangye/USA_gait/video/train' -t 'train' -n 3`

`CUDA_VISIBLE_DEVICES=0 python pose_estimation.py -d '/home/a524wangye/USA_gait/video/val' -t 'val' -n 3`


`CUDA_VISIBLE_DEVICES=0 python pose_estimation.py -d '/home/a524wangye/USA_gait/video/test' -t 'test' -n 3`

then it will generate pose-data of val-set and train-set and save as tfrecords.
you must set the second parameter,when you want to process the train data in order to get the pose-vector, the second parameter should be set 'train'.in a similar way

valset - 'val' ; testset - 'test'
After above, it will save the pose data as tfrecords file which are located
in './TFdata/'
## **To prevent errors, delete data of 'image/frames_data/*' and './TFdata/*' before running pose_estimation.py**

2. run `python run_gait.py -h` to train and validate data. 
   such as :

        usage: run_gait.py [-h] [-n NUM_CLASSES] [-t TRAIN_STEPS] [-e EPOCH]
                   [-b BATCH] [-tr TRAIN_SET] [-va VAL_SET] [-c CHECKPOINT]
                   [-l LEARN_RATE] [-nt NUM_TRAINSET] [-nv NUM_VALSET]

        provide data to train and val

        optional arguments:
        -h, --help            show this help message and exit
        -n NUM_CLASSES, --num_classes NUM_CLASSES
                                number of classes
        -t TRAIN_STEPS, --train_steps TRAIN_STEPS
                                number of iterations per epoch
        -e EPOCH, --epoch EPOCH 
                                number of epoches
        -b BATCH, --batch BATCH
                                number of batch
        -tr TRAIN_SET, --train_set TRAIN_SET
                                path of train tfrecords
        -va VAL_SET, --val_set VAL_SET
                                path of val tfrecords
        -c CHECKPOINT, --checkpoint CHECKPOINT
                                path of saving model
        -l LEARN_RATE, --learn_rate LEARN_RATE
                                learn_rate
        -nt NUM_TRAINSET, --num_trainset NUM_TRAINSET
                                number of train samples ,used to shuffle
        -nv NUM_VALSET, --num_valset NUM_VALSET
                                number of val samples ,used to shuffle

the correct example:

`CUDA_VISIBLE_DEVICES=0 python run_gait.py -n 3 -t 50 -e 10 -b 3 -tr '/home/a524wangye/USA_gait/TFdata/train.tfrecords' -va '/home/a524wangye/USA_gait/TFdata/val.tfrecords' -c '/home/a524wangye/USA_gait/Our_model/' -l 0.001 -nv 3`

After 2, our model will be saved in './Our_model'.

3. After 1 and 2 , we can use the model to test our data.<br>run `python test_model.py -h `
   
        usage: test_model.py [-h] [-m MODEL_FILE] [-t TEST_TFRECORDS] [-n NUM_TEST]

        test model to get acc and loss

        optional arguments:
        -h, --help            show this help message and exit
        -m MODEL_FILE, --model_file MODEL_FILE
                                model path
        -t TEST_TFRECORDS, --test_tfrecords TEST_TFRECORDS
                                test data path
        -n NUM_TEST, --num_test NUM_TEST
                                number of test samples

the correct example
<br>
`CUDA_VISIBLE_DEVICES=0 python test_model.py -m '/home/a524wangye/USA_gait/Our_model/' -t '/home/a524wangye/USA_gait/TFdata/test.tfrecords' -n 3
`



## Data Format

1. raw video data:
    * the name of video is set to 'label_number.mp4'
      * 'label' is label of this video,'_' is used to separate,'number' is the sequence number of the video,such as<br>
        1_0.mp4,1_1.mp4,2_1.mp4 and so on.
        The program will split by '_' to get the label of the video.
    * folder format
      * ![](./temp/construct.png)
      * Set up three folders(train,test,val) to store different data
      
2. frames data
   * **The program is automatically generated without human intervention**
   * Store the frame data corresponding to the above video
   * ![](./temp/2.png)
3. tfrecords data
   * pose_estimation.py will convert the frames data into pose-vector and store them as tfrecords file.
   
   * ![](./temp/5.png)

## Environment configuration
1. Linux 
2. tensorflow-gpu 1.10.0
3. python 3.6
4. opencv 2.4.8
b
