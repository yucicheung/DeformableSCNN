#!~/anaconda2/bin/python
# -*- coding: utf-8 -*-
'''
用读取rec文件的方式用ImageIter处理图片
'''
import mxnet as mx
import os
from nets import * #nets_add1
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

def resume_training():
    '''
    if there are params files, continue training
    if no ,pass
    :return:
    '''
    pass

context = input('set the device to train on(mx.cpu(),mx.gpu()):')
batch_size = 100
data_shape = (3,24,24)
rec_path = os.path.join(os.getcwd(),'rec')
save_step = 5
net_names = ['vanilla_cnn','tailored_cnn',
                 'deformable_vanilla_cnn','deformable_tailored_cnn',]
print net_names
net_name = net_names[int(raw_input('choose from above the net index 1-4 you want to train:'))-1]
num_epoch = int(raw_input('set num of epochs:'))
model_prefix = net_name
VIS = False
log_dir = os.path.join(os.getcwd(),'logs',net_name)
log_file = 'train_'+time.strftime('%m%d_%H_%M',time.localtime())+'.txt'
# create logging file
change_dir(log_dir)
f = open(log_file,'wt')

#------------------------Image Data Iter--------------------------
# define train and val iterator
train_iter = mx.io.ImageRecordIter(path_imgrec=os.path.join(rec_path,'train.rec'),
                                   batch_size=batch_size,
                                   data_shape=data_shape,
                                   shuffle=True,
                                   rand_crop=True,
                                   rand_mirror=True,)
# print train_iter.provide_data,train_iter.provide_label,type(train_iter.provide_data),type(train_iter.provide_label)
val_iter = mx.io.ImageRecordIter(path_imgrec=os.path.join(rec_path,'val.rec'),
                                 batch_size=batch_size,
                                 data_shape=data_shape,
                                 center_crop=True,)

test_iter = mx.io.ImageRecordIter(path_imgrec=os.path.join(rec_path,'test.rec'),
                                  batch_size=batch_size,
                                  data_shape=data_shape,
                                  center_crop=True,)

#---------------------NDArray Data Iter-----------------------------------------



if VIS:
    configs = ['train','val','test']
    for config in configs:
        eval(config+'_iter').reset()
        batch = eval(config + '_iter').next()
        # 索引是0,是为了对应一个label对多个data的情况,返回是一个batch的图片
        data = batch.data[0]
        for i in xrange(10):
            plt.subplot(2,5,i+1)
            plt.imshow(data[i].asnumpy().astype(np.uint8).transpose((1,2,0)))
        plt.show()
# define module
net = eval(net_name)()
mod = mx.mod.Module(symbol=net,
                    context=context,
                    data_names=['data'],
                    label_names=['softmax_label'],)

# allocate memory given the input data and label shapes
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
# initialize parameters by uniform random numbers
mod.init_params(initializer=mx.init.Uniform(scale=.01))
# use SGD with learning rate 0.1 to train
mod.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate': 0.1,})# 'momentum':0.9,})
# mod.init_optimizer(optimizer='adam', optimizer_params={'learning_rate': 0.01, })

# use accuracy as the metric
metric = mx.metric.create('acc')

for epoch in tqdm(xrange(num_epoch),desc=net_name+' training:'):
    start_time = time.time()
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
        mod.forward(batch, is_train=True)       # compute predictions
        mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
        mod.backward()                          # compute gradients
        mod.update()                            # update parameters
    print >>f,'\nEpoch %d, Training %s,Time %.2fs' % (epoch+1, metric.get(),time.time()-start_time),
    for config in['train','val','test']:
        score = mod.score(eval('%s_iter'%config), ['acc'])
        print >>f,"%s Acc: %.4f " % (config,score[0][1]),
    if (epoch+1)%save_step == 0:
        mx.model.save_checkpoint(model_prefix,epoch+1,mod.symbol,mod.get_params()[0],mod.get_params()[1])
f.close()