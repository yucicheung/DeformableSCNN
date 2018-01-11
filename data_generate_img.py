#!~/anaconda2/bin/python
# -*- coding: utf-8 -*-
'''
读取cifar10数据集并且生成NCHW格式NDArray
optional:
1.保存jpg图片
2.生成rec文件
3.可视化部分图片
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from nets import change_dir

def unpickle(fname):
    '''
    load file from pickle file
    param
    -----
        fname: cifar-gz file path
    return
    ------
        dict: original cifar-dict file
    '''
    import cPickle
    with open(fname, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def get_cifar(dir_name):
    '''
    devide cifar dict into train/val/test part
    reshape cifar in shape of NCHW
    param
    ____
        dir_name: directory of cifar folder
    return
    -----
        dict: dict containing the train/val/test data and labels
    '''
    path = os.path.join(dir_name,'cifar-10-batches-py')
    dict = {}
    # load training part
    fname = os.path.join(path,'data_batch_'+str(1))
    load_dict = unpickle(fname)
    data = load_dict['data']
    labels = np.array(load_dict['labels'])

    for i in range(2,5):
        fname = os.path.join(path, 'data_batch_' + str(i))
        load_dict = unpickle(fname)
        data = np.vstack((data,load_dict['data']))
        labels = np.hstack((labels,np.array(load_dict['labels'])))
    dict['train_data'] = data.reshape((-1,3,32,32))
    dict['train_labels'] = labels

    # load val part
    fname = os.path.join(path, 'data_batch_' + str(5))
    load_dict = unpickle(fname)
    data = load_dict['data']
    labels = np.array(load_dict['labels'])
    dict['val_data'] = data.reshape((-1,3,32,32))
    dict['val_labels'] = labels

    # load test part
    fname = os.path.join(path, 'test_batch')
    load_dict = unpickle(fname)
    data = load_dict['data']
    labels = np.array(load_dict['labels'])
    dict['test_data'] = data.reshape((-1,3,32,32))
    dict['test_labels'] = labels
    return dict

def save_pics(dict,img_dir):
    '''
    save pics in jpeg format under folder img_dir
    param
    -----
        dict: cifar dict
        img_dir: dir where pics are gonna be saved
    '''
    change_dir(img_dir)
    configs = ['train','val','test']
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck',]
    for config in configs:
        load_zip = zip(dict[config+'_data'].transpose((0,2,3,1))
                       ,dict[config+'_labels'])
        load_zip = sorted(load_zip,key=lambda x:x[1])
        change_dir(os.path.join(img_dir,config))
        for i in xrange(len(load_zip)):
            change_dir(os.path.join(img_dir,config,label_names[load_zip[i][1]]))
            name = str(i)+'_'+label_names[load_zip[i][1]]+'.jpg'
            cv2.imwrite(name,load_zip[i][0])

def gen_rec(MXNET_root,rec_dir,img_dir):
    '''
    save .rec and .lst files in ./data folder
    param
    -----
        MXNET_root : root folder of MXNET im2rec.py file
        rec_dir: prefix of rec file
        img_dir: dir where the imgs are saved
    '''
    configs = ['train','val','test']
    change_dir(rec_dir)
    for config in configs:
        os.chdir(img_dir)
        os.system('python %s/tools/im2rec.py --list=1 --recursive=1 --shuffle=1 %s %s'
                  %(MXNET_root,os.path.join(rec_dir,config),os.path.join(img_dir,config)))
        os.system('python %s/tools/im2rec.py --num-thread=4 --pass-through=1 %s %s'
                  %(MXNET_root,os.path.join(rec_dir,config),os.path.join(img_dir,config)))

if __name__ == '__main__':
    cifar_dir = '/home/yucicheung/PycharmProjects/mxnetproj/Deformable_SCNN/CIFAR'
    MXNET_root = '/home/yucicheung/anaconda2/lib/python2.7/site-packages/mxnet'
    rec_dir = os.path.join(os.getcwd(),'rec')
    img_dir = os.path.join(os.getcwd(),'img')

    SAVE_PIC = True # whether to save pics
    GEN_REC = True # whether to generate rec file
    VIS = True # whether to visualize some pics

    cifar_dict = get_cifar(cifar_dir)

    if SAVE_PIC:
        # save cifar set in jpeg format
        save_pics(cifar_dict,img_dir)

    if GEN_REC:
        # generate rec and list files
        gen_rec(MXNET_root,rec_dir,img_dir)

    if VIS:
        # Visualize the pics
        configs = ['train','val','test']
        for config in configs:
            for i in xrange(4):
                key = config+'_data'
                pic = cifar_dict[key][i,:,:,:].transpose((1,2,0))
                plt.subplot(1,4,i+1)
                plt.imshow(pic)
            plt.show()
