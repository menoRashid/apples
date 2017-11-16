import numpy as np
import util
# import visualize
import glob
import os
import scipy.misc
import scipy.ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.decomposition
import visualize
import cPickle as pickle
from trying_something_simple import sort_files,get_mask
import random
import itertools

def make_train_test_split():
    data_dir_meta = '../data'
    
    out_dir_split = os.path.join(data_dir_meta,'train_test_split')
    util.mkdir(out_dir_split)
    
    all_files = glob.glob(os.path.join(data_dir_meta,'npy','*','*.npy'))
    all_files_dict = sort_files(all_files)

    num_to_include = [str(num_one)+str(num_two) for num_one in range(1,4) for num_two in range(1,4)]
    
    
    for split_num, num_to_include in enumerate(num_to_include):
        out_file_train = os.path.join(out_dir_split,'train_'+str(split_num)+'.txt')
        out_file_test = os.path.join(out_dir_split,'test_'+str(split_num)+'.txt')
        train_files = []
        test_files = []

        for key_curr in all_files_dict:
            # print key_curr, len(all_files_dict[key_curr])
            if key_curr =='CAD':
                remaining_files= []
                test_files_neg = []
                tree_num = num_to_include[-1]
                for file_curr in all_files_dict[key_curr]:
                    file_num = os.path.split(file_curr)[1]
                    file_num = int(file_num.split('_')[0][3:])
                    if file_num==int(tree_num):
                        test_files_neg.append(file_curr)
                    else:
                        remaining_files.append(file_curr)
                
                random.shuffle(remaining_files)
                test_files.extend(test_files_neg)
                test_files.extend(remaining_files[:6-len(test_files_neg)])
                train_files.extend(remaining_files[6-len(test_files_neg):])

            else:
                for file_curr in all_files_dict[key_curr]:
                    file_num = os.path.split(file_curr)[1]
                    file_num = file_num.split('_')[0][-2:]
                    if file_num==num_to_include:
                        test_files.append(file_curr)
                    else:
                        train_files.append(file_curr)

        print len(train_files)
        print len(test_files)
        assert len(train_files)+len(test_files)==len(set(train_files+test_files))
        print test_files
        util.writeFile(out_file_train,train_files)
        util.writeFile(out_file_test,test_files)

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict


def get_model_sequential(num_classes):
    model = nn.Sequential(OrderedDict([
              ('pool1',nn.AvgPool2d(5, 2,padding = 2)),
              # ('conv3', nn.Conv2d(240, 16, 5, stride = 5)),
              ('conv1', nn.Conv2d(240, 64, 11, stride = 5, padding = 5)),
              ('relu1', nn.ReLU()),
              # ,
              # ('conv2', nn.Conv2d(64, 32, 5, stride = 2, padding = 2)),
              # ('relu2', nn.ReLU()),
              # ('pool2',nn.AvgPool2d(2, 2)),
              ('convt1', nn.ConvTranspose2d(64, num_classes, 5, stride = 5)),
              # ('convt4', nn.ConvTranspose2d(16, 1, 2, stride = 2)),
              # ('conv3', nn.ConvTranspose2d(64, 1, 11, stride = 5)),
              ('upsample', nn.Upsample(size=[250,1600],mode='bilinear')),
              ('prediction', nn.LogSoftmax())
            ]))
    return model




def get_new_batch(train_files_iterator,batch_size,mean_data,pos_neg=True):
    
    classes = ['A','B','C','D']
    all_data = []
    labels = []

    for im_num in range(batch_size): 
        train_file_curr = train_files_iterator.next()
        data_curr = np.load(train_file_curr)
        bin_mask = get_mask(data_curr)

        data_curr = data_curr-mean_data
        all_data.append(data_curr)

        class_curr = os.path.split(train_file_curr)[1].split('_')[0][2]
        if pos_neg:
            if class_curr==classes[-1]:
                label_curr = 2
            else:
                label_curr = 1
        else:
            label_curr = classes.index(class_curr)+1
        
        # print train_file_curr, label_curr
        label_mask = np.zeros(bin_mask.shape)
        label_mask[bin_mask]=label_curr
        labels.append(label_mask)

    all_data = np.array(all_data)
    labels = np.array(labels,dtype=int)
    
    
    return all_data,labels

def save_mean_files():
    data_dir_meta = '../data'
    out_dir_split = os.path.join(data_dir_meta,'train_test_split')
    for split_num in range(9):
        train_file = os.path.join(out_dir_split,'train_'+str(split_num)+'.txt')
        out_file_mean = os.path.join(out_dir_split,'mean_'+str(split_num)+'.npy')

        train_files = util.readLinesFromFile(train_file)
        for idx_train_file,train_file in enumerate(train_files):
            if idx_train_file==0:
                total = np.load(train_file)
            else:
                total = total+np.load(train_file)
            # print np.min(total), np.max(total)
        # print float(len(train_files))
        mean = total/float(len(train_files))
        print mean.shape, np.min(mean), np.max(mean)
        np.save(out_file_mean,mean)

def get_class_weights(train_files,pos_neg, background = 2):
    # classes = 
    # util.readLinesFromFile(train_file)
    classes = [os.path.split(file_curr)[1].split('_')[0][2] for file_curr in train_files]
    print classes
    counts = [classes.count(class_curr) for class_curr in sorted(list(set(classes)))]
    
    print counts

    if pos_neg:
        counts = [sum(counts[:-1])]+[counts[-1]]

    counts = np.array([2*sum(counts)]+counts)
    counts = counts/float(np.sum(counts))
    counts = 1./counts
    counts = counts/float(np.sum(counts))
    # counts = np.array([0.]+list(counts))
    
    return counts

def train_model(train_file,
                test_file,
                mean_file,
                pos_neg,
                batch_size,
                num_epochs,
                save_after,
                disp_after,
                plot_after,
                res_model,
                out_dir_train,lr,dec_after,
                model_num=1):

    util.mkdir(out_dir_train)
    log_file = os.path.join(out_dir_train,'log.txt')
    plot_file = os.path.join(out_dir_train,'loss.jpg')
    log_arr = []
    plot_arr = [[],[]]

    

    num_classes = 3 if pos_neg else 5
    train_files = util.readLinesFromFile(train_file)
    
    class_weights = get_class_weights(train_files, pos_neg, background = 2)
    print class_weights
    
    mean_data = np.load(mean_file)
    epoch_size = len(train_files)/batch_size
    num_iterations = epoch_size*num_epochs
    save_after = save_after*epoch_size
    step_size = dec_after*epoch_size

    print 'epoch_size',epoch_size, 'num_iterations', num_iterations

    torch.cuda.device(0)
    iter_begin = 0
    if res_model == None:
        if model_num==1:
            model = get_model_sequential(num_classes)
        elif model_num==2:
            model = get_model_sequential_2(num_classes)
        elif model_num==3:
            model = get_model_sequential_3(num_classes)
        elif model_num=='3a':
            model = get_model_sequential_3a(num_classes)
        elif model_num=='2a':
            model = get_model_sequential_2a(num_classes)
        elif model_num==4:
            model = get_model_sequential_4(num_classes)
        model = model.cuda()
    else:
        model = torch.load(res_model)
        iter_begin = int(os.path.split(res_model)[1].split('_')[1][:-3])+1
    print model
    optimizer = optim.Adam(model.parameters(),lr = lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size)

    loss_layer = nn.NLLLoss2d(weight = torch.FloatTensor(class_weights).cuda())
    random.shuffle(train_files)
    train_files_iterator = itertools.cycle(train_files)    
    
    for num_iter in range(iter_begin,num_iterations):
        if num_iter % epoch_size ==0 and num_iter>0:
            print 'shuffling'
            random.shuffle(train_files)
            train_files_iterator = itertools.cycle(train_files)    
        
        if num_iter % save_after ==0 or num_iter==num_iterations-1:
            out_file = os.path.join(out_dir_train,'model_'+str(num_iter)+'.pt')
            print 'saving',out_file
            torch.save(model,out_file)
            # save here

        if (num_iter % plot_after==0 and num_iter>0) or num_iter==num_iterations-1:
            util.writeFile(log_file, log_arr)
            visualize.plotSimple([(plot_arr[0],plot_arr[1])],out_file = plot_file,title = 'Loss',xlabel = 'Iteration',ylabel = 'Loss')

        if num_iter %step_size==0 and num_iter>0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] *0.1

        data, labels = get_new_batch(train_files_iterator, batch_size, mean_data,pos_neg)        
        data = Variable(torch.FloatTensor(data).cuda())
        labels = Variable(torch.LongTensor(labels).cuda())
        optimizer.zero_grad()
        loss = loss_layer(model(data), labels)    
        loss_iter = loss.data[0]
        loss.backward()
        optimizer.step()

        plot_arr[0].append(num_iter); plot_arr[1].append(loss_iter)
        str_display = 'lr: %.6f, iter: %d, loss: %.4f' %(optimizer.param_groups[0]['lr'],num_iter,loss_iter)
        log_arr.append(str_display)
        if num_iter %disp_after ==0 and num_iter>0:
            print str_display

            


def test_model(test_file,mean_file,out_dir_train,model_num,pos_neg):
    res_model = os.path.join(out_dir_train,'model_'+str(model_num)+'.pt')
    out_dir_test = os.path.join(out_dir_train,'results')
    util.mkdir(out_dir_test)

    mean_data = np.load(mean_file)

    model = torch.load(res_model)
    model.eval()
    # out_dir_test = os.path.join(out_dir_train,'results_noEval')
    

    batch_size = 1
    test_files = util.readLinesFromFile(test_file)
    num_iterations = len(test_files)

    test_files_iterator = itertools.cycle(test_files)    
    
    max_val = 2 if pos_neg else 4

    for num_iter in range(num_iterations):
        data, labels = get_new_batch(test_files_iterator, batch_size, mean_data,pos_neg)        
        data = Variable(torch.FloatTensor(data).cuda())
        labels = Variable(torch.LongTensor(labels).cuda())
        prediction = model(data)
        prediction = np.argmax(prediction.data.cpu().numpy(),1)
        labels = labels.data.cpu().numpy()
        print prediction.shape
        print np.min(prediction),np.max(prediction)
        print np.min(labels),np.max(labels)
        out_file_gt = os.path.join(out_dir_test, str(num_iter)+'_gt.jpg')
        out_file_pred = os.path.join(out_dir_test,str(num_iter)+'_pred.jpg')

        # scipy.misc.imsave(out_file_pred,prediction[0]*255./float(max_val))
        scipy.misc.toimage(prediction[0], cmin=0, cmax=max_val).save(out_file_pred)
        # scipy.misc.imsave(out_file_gt,labels[0]*255./float(max_val))
        scipy.misc.toimage(labels[0], cmin=0, cmax=max_val).save(out_file_gt)

        visualize.writeHTMLForFolder(out_dir_test,height=125,width = 800)




def script_train_test():
    data_dir_meta = '../data'
    out_dir_split = os.path.join(data_dir_meta,'train_test_split')
    
    split_num = '0'
    train_file = os.path.join(out_dir_split,'train_'+split_num+'.txt')
    test_file = os.path.join(out_dir_split,'test_'+split_num+'.txt')
    # train_file
    mean_file = os.path.join(out_dir_split,'mean_'+str(split_num)+'.npy')

    pos_neg = False
    batch_size = 5
    num_epochs = 40
    save_after = 5
    disp_after = 1
    plot_after = 10
    lr = 0.001
    dec_after = 20
    res_model =  None
    # '../experiments/simple_False_0/model_150.pt'
    folder_arr = [str(val) for val in [pos_neg,split_num,num_epochs,lr,dec_after]]
    model_num = 4
    # '2a'
    # out_dir_train = '../experiments/simple_zero_background_weights_'+'_'.join(folder_arr)
    out_dir_train = '../experiments/model_'+str(model_num)+'_regular_weights_'+'_'.join(folder_arr)

    # folder_arr = [str(val) for val in [pos_neg,split_num,40,lr,40]]
    # out_dir_res = '../experiments/model_'+str(model_num)+'_regular_weights_'+'_'.join(folder_arr)
    # res_model = os.path.join(out_dir_res,'model_225.pt')
    # out_dir_train = '../experiments/model_2_regular_weights_res_no_dip_'+'_'.join(folder_arr)
    
    train_model(train_file,
                test_file,
                mean_file,
                pos_neg,
                batch_size,
                num_epochs,
                save_after,
                disp_after,
                plot_after,
                res_model,
                out_dir_train,
                lr,dec_after, model_num)

    model_num = 599
    test_model(test_file,mean_file,out_dir_train,model_num,pos_neg)


def get_model_sequential_2(num_classes):
    print '2'
    model = nn.Sequential(OrderedDict([
              # ('pool1',nn.AvgPool2d(5, 2,padding = 2)),
              # ('conv3', nn.Conv2d(240, 16, 5, stride = 5)),
              ('conv1', nn.Conv2d(240, 64, 11, stride = 5, padding = 5)),
              ('relu1', nn.ReLU()),
              # ,
              ('conv2', nn.Conv2d(64, 32, 5, stride = 2, padding = 2)),
              ('relu2', nn.ReLU()),

              ('conv3', nn.Conv2d(32, num_classes, 1, stride = 1, padding = 0)),
              # ('relu3', nn.ReLU()),
              # ('pool2',nn.AvgPool2d(2, 2)),
              # ('convt1', nn.ConvTranspose2d(64, num_classes, 5, stride = 5)),
              # ('convt4', nn.ConvTranspose2d(16, 1, 2, stride = 2)),
              # ('conv3', nn.ConvTranspose2d(64, 1, 11, stride = 5)),
              ('upsample', nn.Upsample(size=[250,1600],mode='bilinear')),
              ('prediction', nn.LogSoftmax())
            ]))
    return model

def get_model_sequential_2a(num_classes):
    print '2a'
    model = nn.Sequential(OrderedDict([
              ('conv1', nn.Conv2d(240, 64, 11, stride = 5, padding = 5)),
              ('relu1', nn.ReLU()),
              ('bn1', nn.BatchNorm2d(64)),
              
              ('conv2', nn.Conv2d(64, 32, 5, stride = 2, padding = 2)),
              ('relu2', nn.ReLU()),
              ('bn2', nn.BatchNorm2d(32)),
              
              ('conv3', nn.Conv2d(32, num_classes, 1, stride = 1, padding = 0)),
              
              ('upsample', nn.Upsample(size=[250,1600],mode='bilinear')),
              ('prediction', nn.LogSoftmax())
            ]))
    return model

def get_model_sequential_3(num_classes):
    print '3'
    model = nn.Sequential(OrderedDict([
              # ('pool1',nn.AvgPool2d(5, 2,padding = 2)),
              # ('conv3', nn.Conv2d(240, 16, 5, stride = 5)),
              ('conv1', nn.Conv2d(240, 64, 5, stride = 2, padding = 2)),
              ('relu1', nn.ReLU()),
              # ,
              ('conv2', nn.Conv2d(64, 32, 5, stride = 2, padding = 2)),
              ('relu2', nn.ReLU()),

              ('conv3', nn.Conv2d(32, 16, 5, stride = 2, padding = 2)),
              ('relu3', nn.ReLU()),
              ('conv4', nn.Conv2d(16, num_classes, 1, stride = 1, padding = 0)),
              # ('pool2',nn.AvgPool2d(2, 2)),
              # ('convt1', nn.ConvTranspose2d(64, num_classes, 5, stride = 5)),
              # ('convt4', nn.ConvTranspose2d(16, 1, 2, stride = 2)),
              # ('conv3', nn.ConvTranspose2d(64, 1, 11, stride = 5)),
              ('upsample', nn.Upsample(size=[250,1600],mode='bilinear')),
              ('prediction', nn.LogSoftmax())
            ]))
    return model

def get_model_sequential_3a(num_classes):
    print '3a'
    model = nn.Sequential(OrderedDict([
              # ('pool1',nn.AvgPool2d(5, 2,padding = 2)),
              # ('conv3', nn.Conv2d(240, 16, 5, stride = 5)),
              ('conv1', nn.Conv2d(240, 64, 5, stride = 2, padding = 2)),
              ('relu1', nn.ReLU()),
              # ,
              ('conv2', nn.Conv2d(64, 32, 5, stride = 2, padding = 2)),
              ('relu2', nn.ReLU()),

              ('conv3', nn.Conv2d(32, 16, 5, stride = 2, padding = 2)),
              ('relu3', nn.ReLU()),
              ('dropout3',torch.nn.Dropout2d(p=0.5)),

              ('conv4', nn.Conv2d(16, num_classes, 1, stride = 1, padding = 0)),
              # ('pool2',nn.AvgPool2d(2, 2)),
              # ('convt1', nn.ConvTranspose2d(64, num_classes, 5, stride = 5)),
              # ('convt4', nn.ConvTranspose2d(16, 1, 2, stride = 2)),
              # ('conv3', nn.ConvTranspose2d(64, 1, 11, stride = 5)),
              ('upsample', nn.Upsample(size=[250,1600],mode='bilinear')),
              ('prediction', nn.LogSoftmax())
            ]))
    return model

def get_model_sequential_4(num_classes):
    model = nn.Sequential(OrderedDict([
              ('pool1',nn.AvgPool2d(5, 2,padding = 2)),
              ('conv1', nn.Conv2d(240, 64, 11, stride = 5, padding = 5)),
              ('relu1', nn.ReLU()),
              ('bn1', nn.BatchNorm2d(64)),
              
              ('convt1', nn.ConvTranspose2d(64, 32, 5, stride = 5)),
              ('relut1', nn.ReLU()),
              ('bnt1', nn.BatchNorm2d(32)),
              
              ('conv3', nn.Conv2d(32, num_classes, 1, stride = 1, padding = 0)),
              ('upsample', nn.Upsample(size=[250,1600],mode='bilinear')),
              ('prediction', nn.LogSoftmax())
            ]))
    return model   


def main():
    script_train_test()

    return
    data = np.zeros((240,250,1600))
    labels = np.zeros((250,1600),dtype = int)
    # np.load(train_files[0])
    num_classes = 5
    print data.shape
    print np.min(data), np.max(data)
    net = get_model_sequential_4(num_classes)
    print net
    torch.cuda.device(0)

    net = net.cuda()
    data = Variable(torch.FloatTensor(data[np.newaxis,:,:,:]).cuda())
    labels = Variable(torch.LongTensor(labels[np.newaxis,:,:]).cuda())

    
    # loss_layer = nn.NLLLoss2d(weight = torch.FloatTensor(class_weights).cuda())
    prediction = net(data)
    print prediction.data.shape
    # print labels.data.shape

    # loss = loss_layer(net(data), labels)
    # print loss.data
    
    # net.zero_grad()

    # loss.backward()

    # # train_files = util.readLinesFromFile(train_file)
    # # test_files = util.readLinesFromFile(test_file)
    # return
    # data = np.zeros((240,250,1600))
    # # np.load(train_files[0])
    # print data.shape
    # print np.min(data), np.max(data)
    # net = get_model_sequential()
    # print net
    # torch.cuda.device(0)

    # net = net.cuda()
    # data = Variable(torch.FloatTensor(data[np.newaxis,:,:,:])).cuda()
    # output = net.forward(data)
    # print output.data.shape
    # print net.conv1



if __name__=='__main__':
    main()




