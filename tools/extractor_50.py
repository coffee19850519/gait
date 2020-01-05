import os
import shutil

def bianli(dir_):

    # fyc
    list_dirs = os.listdir(dir_)
    count = 0
    for dirs in list_dirs:
        list_sub_dirs = os.listdir(dir_+str(dirs))
        # 00_*
        for sub_dir in list_sub_dirs:
            zero_dir_list = os.listdir(dir_+str(dirs)+"/"+str(sub_dir))
            if sub_dir in ['90_1','90_2','90_3','90_4'] and  len(zero_dir_list)>50:
                print(str(dirs)+"/"+str(sub_dir)+"  "+str(len(zero_dir_list)))
                count+=1
    # print(count)
                extractor(dir_+str(dirs)+"/"+str(sub_dir),str(dirs),str(sub_dir))
                # 将图片数多于50的提取出来
                

def extractor(dir_,first_part,second_part):
    temp = "/media/wang/13c7394c-f459-4ae8-93cf-2c8095319705/DatasetA/new_gait_data_45_view/"


    # 移动前50张
    for i in range(1,51):
        if i<10:
            if not os.path.exists(temp+str(first_part)+"/"+str(second_part)):
             
                os.makedirs(temp+str(first_part)+"/"+str(second_part))
            shutil.copy(dir_+"/"+str(first_part)+"-"+str(second_part)+"-"+"00"+str(i)+".png",temp+str(first_part)+"/"+str(second_part)+"/"+str(first_part)+"-"+str(second_part)+"-"+"00"+str(i)+".png")
        else:
            if not os.path.exists(temp+str(first_part)+"/"+str(second_part)):
                
                os.makedirs(temp+str(first_part)+"/"+str(second_part))
            shutil.copy(dir_+"/"+str(first_part)+"-"+str(second_part)+"-"+"0"+str(i)+".png",temp+str(first_part)+"/"+str(second_part)+"/"+str(first_part)+"-"+str(second_part)+"-"+"0"+str(i)+".png")


# def generate_val(dir_):
#     list_dirs = os.listdir(dir_)
#     count = 0
#     for dirs in list_dirs:
#         # print(dirs)
#         # 移动即可
#         for i in range(1,51):
#             # if not os.path.exists(dir_)   
#         shutil.move(dir_+str(dirs)+"/"+"00_3"+"/"+str(i)+)


dir_='/media/wang/13c7394c-f459-4ae8-93cf-2c8095319705/DatasetA/gaitdb/'
# generate_val(dir_)
# extractor(dir_)
bianli(dir_)