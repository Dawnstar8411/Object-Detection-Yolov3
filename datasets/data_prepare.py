import os
import random

if __name__ == "__main__":
    filepath = '/home/deep/chen/Datasets/Pineapple/train/images'
    filelist = os.listdir(filepath)
    N = len(filelist)
    train_index_list = random.sample(range(1,N),N*0.8)
    train_txt = open('/home/deep/chen/Datasets/Pineapple/train/train.txt','w')
    valid_txt = open('/home/deep/chen/Datasets/Pineapple/train/valid.txt','w')
    for i,f in enumerate(filelist):
        filename = '/home/deep/chen/Datasets/Pineapple/train/images/'+f
        if i in train_index_list:
            train_txt.write(filename)
            train_txt.write('\n')
        else:
            valid_txt.write(filename)
            valid_txt.write('\n')
    train_txt.close()
    valid_txt.close()



