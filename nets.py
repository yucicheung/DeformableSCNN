#!~/anaconda2/bin/python
# -*- coding: utf-8 -*-

'''
包含所有需要用到的网络结构
以及对结构进行可视化和保存
notes:
1.仅就这个网络,vanilla与tailored cnn的区别为是否禁用bias
2.deformable只在GPU上可执行
'''
import mxnet as mx
import os

def convfactory(data,num_filter,kernel,suffix,no_bias,stride=(1,1),pad=(0,0),):
    '''build convnet and activate with ReLU'''
    net = mx.sym.Convolution(
        data=data, num_filter=num_filter, kernel=kernel,no_bias=no_bias,
        stride=stride, pad=pad,name='Conv_%s'%suffix)
    net = mx.sym.Activation(data=net,act_type='relu',name='relu_%s'%suffix)
    return net

def easypooling(data,suffix,pool_type='avg',kernel=(2,2),stride=(2,2),pad=(0,0)):
    '''set some paramerers as default value so the calling is easier'''
    net = mx.sym.Pooling(
        data=data, kernel=kernel, stride=stride,
        pad=pad, pool_type=pool_type, name='Pooling_%s'%suffix)
    return net

def deformconvfactory(data,num_filter,kernel,suffix,deform_pad,no_bias,
                      dilate=(1,1),pad=(0,0),num_deform_group=1,stride=(1,1)):
    '''build deformable convnet and activate it with ReLU'''
    num_offset_filter = 2 * kernel[0] * kernel[1] * num_deform_group
    offset = mx.sym.Convolution(
        data=data,stride=stride,kernel=kernel,no_bias=no_bias,
        pad=pad,num_filter=num_offset_filter,name='offset_map%s'%suffix
    )
    net = mx.contrib.symbol.DeformableConvolution(
        data=data,offset=offset,dilate=dilate,stride=stride,num_filter=num_filter,
        pad=deform_pad,kernel=kernel,name='Deform_conv%s'%suffix,no_bias=no_bias,)
    net = mx.sym.Activation(data=net,act_type='relu',name='ReLU%s'%suffix)
    return net

def vanilla_cnn():
    '''
    build vanilla CNN
    return:
      symbol that represents vanilla CNN
    '''
    no_bias = False
    in_data = mx.sym.Variable(name='data')
    # build network
    net = convfactory(data=in_data,no_bias=no_bias,num_filter=64,kernel=(5,5),suffix='1',pad=(0,0))
    net = easypooling(data=net,suffix='1')
    net = convfactory(data=net,no_bias=no_bias,num_filter=64,kernel=(5,5),suffix='2',pad=(0,0))
    net = easypooling(data=net,suffix='2')
    net = convfactory(data=net,no_bias=no_bias,num_filter=64,kernel=(3,3),suffix='3',pad=(0,0))
    net = mx.sym.FullyConnected(data=net,num_hidden=64,name='FC1',no_bias=no_bias)
    net = mx.sym.Activation(data=net,act_type='relu',name='relu3')
    net = mx.sym.FullyConnected(data=net,num_hidden=10,name='FC2',no_bias=no_bias)
    net = mx.sym.SoftmaxOutput(data=net,name='softmax')
    return net

def tailored_cnn():
    '''
    build tailored cnn
    return:
      symbol that represents tailored cnn
    '''
    no_bias = True
    in_data = mx.sym.Variable(name='data')
    # build network
    net = convfactory(data=in_data,no_bias=no_bias,num_filter=64,kernel=(5,5),suffix='1',pad=(0,0))
    net = easypooling(data=net,suffix='1')
    net = convfactory(data=net,no_bias=no_bias,num_filter=64,kernel=(5,5),suffix='2',pad=(0,0))
    net = easypooling(data=net,suffix='2')
    net = convfactory(data=net, no_bias=no_bias,num_filter=64,kernel=(3,3),suffix='3',pad=(0,0))
    net = mx.sym.FullyConnected(data=net,num_hidden=64,name='FC1',no_bias=no_bias)
    net = mx.sym.Activation(data=net,act_type='relu',name='relu3')
    net = mx.sym.FullyConnected(data=net,num_hidden=10,name='FC2',no_bias=no_bias)
    net = mx.sym.SoftmaxOutput(data=net,name='softmax')
    return net

def deformable_vanilla_cnn():
    '''
    build deformable vanilla CNN network
    return:
      represents symbol that represents deformable vanilla CNN
    '''
    no_bias = False
    in_data = mx.sym.Variable(name='data')
    # build network
    net = convfactory(data=in_data, no_bias=no_bias, num_filter=64, kernel=(5, 5), suffix='1', pad=(0, 0))
    net = easypooling(data=net, suffix='1')
    net = convfactory(data=net, no_bias=no_bias, num_filter=64, kernel=(5, 5), suffix='2', pad=(0, 0))
    net = easypooling(data=net, suffix='2')
    net = deformconvfactory(data=net,num_filter=64,kernel=(3,3),dilate=(1,1),no_bias=no_bias,
                            num_deform_group=2,suffix='3',deform_pad=(0,0),pad=(0,0))
    net = mx.sym.FullyConnected(data=net,num_hidden=64,name='FC1',no_bias=no_bias)
    net = mx.sym.Activation(act_type='relu',name='ReLU4',data=net)
    net = mx.sym.FullyConnected(data=net,num_hidden=10,name='FC2',no_bias=no_bias)
    net = mx.sym.SoftmaxOutput(data=net,name='softmax')
    return net

def deformable_tailored_cnn():
    '''
    build deformable tailored CNN network
    return:
      represents symbol that represents deformable vanilla CNN
    '''
    no_bias = True
    in_data = mx.sym.Variable(name='data')
    # build network
    net = convfactory(data=in_data, no_bias=no_bias, num_filter=64, kernel=(5, 5), suffix='1', pad=(0, 0))
    net = easypooling(data=net, suffix='1')
    net = convfactory(data=net, no_bias=no_bias, num_filter=64, kernel=(5, 5), suffix='2', pad=(0, 0))
    net = easypooling(data=net, suffix='2')
    net = deformconvfactory(data=net,num_filter=64,kernel=(3,3),dilate=(1,1),no_bias=no_bias,
                            num_deform_group=2,suffix='3',deform_pad=(0,0),pad=(0,0))
    net = mx.sym.FullyConnected(data=net,num_hidden=64,name='FC1',no_bias=no_bias)
    net = mx.sym.Activation(act_type='relu',name='ReLU4',data=net)
    net = mx.sym.FullyConnected(data=net,num_hidden=10,name='FC2',no_bias=no_bias)
    net = mx.sym.SoftmaxOutput(data=net,name='softmax')
    return net

def change_dir(dir_name):
    '''
    ensure the existence of a dir
    and change the working directory to it
    '''
    if os.path.exists(dir_name):
        pass
    else:
        os.makedirs(dir_name)
    os.chdir(dir_name)

if __name__ =='__main__':
    # To visualize the network and save the pic of arch
    net_names = ['vanilla_cnn','tailored_cnn',
                 'deformable_vanilla_cnn','deformable_tailored_cnn',]
    cwd = os.getcwd()
    for net_name in net_names:
        # visualize nets and save pics
        change_dir(os.path.join(cwd,'arch_pic',net_name+'_arch'))
        network = eval(net_name)()
        #mx.viz.plot_network(network,shape={'data':(128,3,24,24)},save_format='jpg',title=net_name).view()
        mx.viz.plot_network(network,shape={'data':(128,3,24,24)},save_format='jpg',title=net_name).render()