#coding=utf-8
import cv2
import os
print(2)
temp="2_3"
# temp1="0_1"此

# 此处的参数可直接替换
# train_dir = '/media/wang/13c7394c-f459-4ae8-93cf-2c8095319705/gait公司数据/gait/train/'
# val_dir = '/media/wang/13c7394c-f459-4ae8-93cf-2c8095319705/gait公司数据/gait/val/'



def video_to_frames(dir_,train_or_val):

    # 读取train下面所有的视频名,视频文件夹名字即为label,故所有的视频文件夹名字均以1 2 形式命名
    if str(dir_)[-1] !='/':
        dir_ = dir_+'/'
    temp = './image/frames_data/'+str(train_or_val)+'/'
    
    video_name_list = os.listdir(dir_)
    print(video_name_list)
    for video in video_name_list:

        video_add = dir_+str(video)
        
        if train_or_val=='train':
            if not os.path.isdir('./image/frames_data/'+str(train_or_val)+'/'+str(str(video).split('.')[0])):
                os.makedirs('./image/frames_data/'+str(train_or_val)+'/'+str(str(video).split('.')[0]))
            
        else:
            if not os.path.isdir('./image/frames_data/'+str(train_or_val)+'/'+str(str(video).split('.')[0])):
                os.makedirs('./image/frames_data/'+str(train_or_val)+'/'+str(str(video).split('.')[0]))
        
        
        vc = cv2.VideoCapture(video_add)
        print(video_add)
        if vc.isOpened():
            rval,frame = vc.read()
        else:
            rval = False
        c = 1
        while rval:
               
            rval,frame=vc.read()
            # 读取到视频后,创建文件夹存储视频帧
            if c==299:
                pass
            else:
                cv2.imwrite('./image/frames_data/'+str(train_or_val)+'/'+str(str(video).split('.')[0])+"/"+str(c)+'.jpg',frame)
            c = c+1
            cv2.waitKey(1)
        vc.release()


    return temp

# if __name__ == "__main__":
#     # video_to_frames(train_dir,'train')
#     a = video_to_frames(val_dir,'val')
#     print(a)