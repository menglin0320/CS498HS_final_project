import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
sample_path='C:/Users/MyLappy/CS 498/.spyproject/train_data.txt'
label_path='C:/Users/MyLappy/CS 498/.spyproject/test_data.txt'


with open(sample_path, 'r') as f1:
    lines=f1.readlines()

data_list=[]
for line in lines:
    tempo1 = line.strip('\n').split('\t')
    data_list.append(tempo1)
#temp2=shuffle(lines2)
n_samples=len(data_list)
data_array=np.array(data_list)
with open(label_path, 'r') as f2:
    lines3=f2.readlines()
label_list=[]
for line in lines3:
    tempo2=line.strip('\n').split('\t')
    label_list.append(tempo2)

label_array = np.array(label_list)
label_array_no_id = label_array[:,1:2]
combine=np.hstack((data_array,label_array_no_id))
np.random.shuffle(combine)
#print(combine2[0][0])

n_training=int(0.8*n_samples)
n_test = n_samples - n_training
trainn=combine[0:n_training,:]
testn=combine[n_training:,:]

text_len_train=np.zeros((trainn.shape[0],1))
text_len_test=np.zeros((testn.shape[0],1))
for i in range(0,trainn.shape[0]):
    text_len_train[i]=trainn[i][1].count(' ')+1
for i in range(0,testn.shape[0]):
    text_len_test[i]=testn[i][1].count(' ')+1
#print(text_len_train[0][0])
trainn=np.hstack((trainn,text_len_train))
testn=np.hstack((testn,text_len_test))
#print(trainn[0][6],trainn[1][6])
trainn = trainn[trainn[:,6].argsort()]
testn = testn[testn[:,6].argsort()]

#print(trainn[0][6],trainn[1][6])
#print(trainn[0][0])


#trainn.sort(key = lambda x: len(x[0]))
#print(trainn[0][:])
#print(trainn[1][:])