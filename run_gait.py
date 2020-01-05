from gait_model import Gait_model

import argparse





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="provide data to train and val")

    # 添加参数
    parser.add_argument('-n','--num_classes',type=int,help="number of classes")
    parser.add_argument('-t','--train_steps',type=int,help="number of iterations per epoch")
    parser.add_argument('-e','--epoch',type=int,help="number of epoches")
    parser.add_argument('-b','--batch',type=int,help="number of batch")
    parser.add_argument('-tr','--train_set',type=str,help="path of train tfrecords")
    parser.add_argument('-va','--val_set',type=str,help="path of val tfrecords")
    parser.add_argument('-c','--checkpoint',type=str,help="path of saving model")
    parser.add_argument('-l','--learn_rate',type=float,help="learn_rate")
    parser.add_argument('-nt','--num_trainset',type=int,help="number of  train samples ,used to shuffle")
    parser.add_argument('-nv','--num_valset',type=int,help="number of  val samples ,used to shuffle")
    
    
    args = parser.parse_args()

    # print("dsadsadsadas")
    gait = Gait_model(num_classes=args.num_classes,train_steps=args.train_steps,epoch=args.epoch,batch=args.batch,\
                train_tfrecord=args.train_set,val_tfrecord=args.val_set,\
                checkpoint=args.checkpoint,learn_rate=args.learn_rate,num_trainset=args.num_trainset,num_valset=args.num_valset)


    gait.train_and_val_model()

