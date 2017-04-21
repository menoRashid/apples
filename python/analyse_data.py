import os;
import util;
import gdal;
import matplotlib.pyplot as plt;
import numpy as np;
import multiprocessing;
import shutil;
import random;

def readBilFile(bil):
    gdal.GetDriverByName('EHdr').Register()
    img = gdal.Open(bil)
    if img is not None:
        data=[];
        for i in range(1,img.RasterCount+1):
            # print i
            band=img.GetRasterBand(i);
            data.append(band.ReadAsArray());
        data =np.array(data);
    else:
        data=None
    return data

def saveAsNpy((in_file,out_file,num)):
    data=readBilFile(in_file);
    if data is not None:
        print num,data.shape
        np.save(out_file,data);
    else:
        print num,data

def saveAsNpyMultiProc(in_folders,out_folder_meta,num_proc=multiprocessing.cpu_count()):
    in_files_all=[];
    out_files_all=[];
    for in_folder_curr in in_folders:
        in_folder_im=os.path.split(in_folder_curr)[-1];
        out_folder_curr=os.path.join(out_folder_meta,in_folder_im);
        util.mkdir(out_folder_curr);
        files=util.getFilesInFolder(in_folder_curr,'.bil');
        files_no_ext=util.getFileNames(files,ext=False);
        in_files=[os.path.join(in_folder_im,file_curr) for file_curr in files];
        out_files=[os.path.join(out_folder_curr,file_curr+'.npy') for file_curr in files_no_ext];
        in_files_all.extend(in_files);
        out_files_all.extend(out_files);

    print len(in_files_all),len(out_files_all);
    p=multiprocessing.Pool(num_proc);
    p.map(saveAsNpy,zip(in_files_all,out_files_all,range(len(in_files_all))));

def saveTrainTestFiles():
    in_folder_meta='/Users/maheenrashid/Dropbox (Personal)/Davis_docs/apples/Dipped Apples_Jul02/npy';
    list_folders = ['CAA_Jul02','CAB_Jul02','CAC_Jul02','CAD_Jul02']
    
    out_file_pos_train=os.path.join(in_folder_meta,'pos_train.txt');
    out_file_neg_train=os.path.join(in_folder_meta,'neg_train.txt');
    out_file_pos_test=os.path.join(in_folder_meta,'pos_test.txt');
    out_file_neg_test=os.path.join(in_folder_meta,'neg_test.txt');

    pos_folders=[os.path.join(in_folder_meta,folder_curr) for folder_curr in list_folders[:3]];
    pos_files=[];
    for pos_folder in pos_folders:
        pos_files.extend(util.getFilesInFolder(pos_folder,'.npy'));
    
    neg_folder=os.path.join(in_folder_meta,list_folders[-1]);
    neg_files=util.getFilesInFolder(neg_folder,'.npy');
    print len(pos_files),len(neg_files),pos_files[0],neg_files[0]
    
    pos_num_train=len(pos_files)-8;
    neg_num_train=len(neg_files)-6;

    random.shuffle(pos_files);
    random.shuffle(neg_files);

    pos_train=pos_files[:pos_num_train];
    neg_train=neg_files[:neg_num_train];
    
    pos_test=pos_files[pos_num_train:];
    neg_test=neg_files[neg_num_train:];
    
    util.writeFile(out_file_pos_train,pos_train);
    util.writeFile(out_file_neg_train,neg_train);
    util.writeFile(out_file_pos_test,pos_test);
    util.writeFile(out_file_neg_test,neg_test);

def getAverageNpy(files):
    for idx,file_curr in enumerate(files):
        data_curr=np.load(file_curr);
        if idx==0:
            total=data_curr;
        else:
            total=total+data_curr;

    average=total/len(files);
    print total.shape,average.shape
    return average

def saveAverageNpy():
    in_folder_meta='/Users/maheenrashid/Dropbox (Personal)/Davis_docs/apples/Dipped Apples_Jul02/npy';
    pos_train=os.path.join(in_folder_meta,'pos_train.txt');
    neg_train=os.path.join(in_folder_meta,'neg_train.txt');
    avg_file=os.path.join(in_folder_meta,'mean_im.npy');
    files=util.readLinesFromFile(pos_train);
    files.extend(util.readLinesFromFile(neg_train));
    print len(files);
    avg=getAverageNpy(files);
    np.save(avg_file,avg);


def main():
    
    

    return
    in_folder_meta='/Users/maheenrashid/Dropbox (Personal)/Davis_docs/apples/Dipped Apples_Jul02';
    list_folders = ['CAA_Jul02','CAB_Jul02','CAC_Jul02','CAD_Jul02']
    in_folders=[os.path.join(in_folder_meta,folder_curr) for folder_curr in list_folders];
    out_folder_meta=os.path.join(in_folder_meta,'npy');
    util.mkdir(out_folder_meta);
    saveAsNpyMultiProc(in_folders,out_folder_meta);


if __name__=='__main__':
    main();


