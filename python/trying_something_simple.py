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

def save_viz(in_file, out_file):
	data = np.load(in_file)
	print data.shape
	data = np.mean(data,0)
	scipy.misc.imsave(out_file,data);
	
def sort_files(all_files):
	dict_classes = {}
	for file_curr in all_files:
		file_name = os.path.split(file_curr)[1]
		pest_class = file_name.split('_')[0][:3]
		if pest_class in dict_classes:
			dict_classes[pest_class].append(file_curr)
		else:
			dict_classes[pest_class] = [file_curr]
	return dict_classes

def get_mask(data,filter_size = 15, thresh = 0.3):
	data = data[1,:,:]
	min_val = np.min(data)
	max_val = np.max(data)
	vals = np.linspace(min_val,max_val,10)[1:-1]
	filter_size = 15
	filter_curr = (1./filter_size**2)*np.ones((filter_size,filter_size))
	out_arr = np.zeros(data.shape)
	out_arr[data>=thresh]=1.
	 
	out_arr = scipy.ndimage.filters.convolve(out_arr, filter_curr)
	out_arr[out_arr<thresh]=0
	out_arr[out_arr>=thresh]=1.
	out_arr = np.logical_not(out_arr)
	return out_arr

def random_pick_features(file_curr,per_im_to_keep):
	data = np.load(file_curr)
	mask = get_mask(data)
	idx_keep = np.where(mask)
	idx_keep = np.vstack(idx_keep).T
	# print idx_keep.shape
	idx_idx_keep = np.random.randint(0,idx_keep.shape[0],per_im_to_keep)
	idx_keep = idx_keep[idx_idx_keep]
	vecs_curr = [data[:,r,c] for r,c in idx_keep]
	return vecs_curr

def get_rand_picked_pca(files_chosen,per_im_to_keep, num_comp_to_keep=None):

	vecs = []
	for file_curr in files_chosen:
		vecs_curr = random_pick_features(file_curr, per_im_to_keep)
		vecs = vecs+vecs_curr

	vecs = np.array(vecs)
	norms = np.linalg.norm(vecs, axis =1, keepdims = True)
	vecs = vecs/norms
	if num_comp_to_keep is None:
		pca = sklearn.decomposition.PCA(whiten=True)
	else:
		pca = sklearn.decomposition.PCA(n_components = num_comp_to_keep, whiten=True)
	pca.fit(vecs)

	return pca

def plot_num_components():
	data_dir = '../data/npy'
	out_dir = '../scratch/viz_thresh'
	util.mkdir(out_dir)

	all_files = glob.glob(os.path.join(data_dir,'*','*.npy'))
	dict_classes = sort_files(all_files)
	files_chosen = [dict_classes[key_curr][0] for key_curr in dict_classes.keys()]

	xAndYs = []
	legend_entries =[]
	for per_im_to_keep in [100,1000,10000]:
		pca = get_rand_picked_pca(files_chosen,per_im_to_keep)
		variance_sum  = np.cumsum(pca.explained_variance_ratio_)
		# print variance_sum
		xAndYs.append((range(len(variance_sum)),variance_sum))
		legend_entries.append(str(per_im_to_keep))
		# plt.figure()
		# plt.plot(range(len(variance_sum)),variance_sum)
		# plt.title(str(per_im_to_keep))
		# plt.xlabel('n_comp')
		# plt.ylabel('variance cumsum')
		# plt.show()
	plt.ion()
	
	visualize.plotSimple(xAndYs,xlabel='n_comp',ylabel='variance cumsum',legend_entries=legend_entries)
	plt.show()
	raw_input()

def save_pca():
	data_dir = '../data/npy'
	out_dir = '../scratch/viz_thresh'
	util.mkdir(out_dir)

	all_files = glob.glob(os.path.join(data_dir,'*','*.npy'))
	dict_classes = sort_files(all_files)
	files_chosen = [dict_classes[key_curr][0] for key_curr in dict_classes.keys()]
	per_im_to_keep = 10000

	pca = get_rand_picked_pca(files_chosen,per_im_to_keep,num_comp_to_keep=100)
	out_file = '../scratch/pca_10000_100.pkl'
	pickle.dump(pca,open(out_file,'wb'))

	# pca = get_rand_picked_pca(files_chosen,per_im_to_keep)
	# components = pca.components_
	# variance = pca.explained_variance_ratio_
	# out_file = '../scratch/pca_10000.npz'
	# np.savez(out_file, components = components, variance = variance)
	

def main():
	data_dir = '../data/npy'
	out_dir = '../scratch/viz_thresh'
	util.mkdir(out_dir)
	out_file = '../scratch/pca_10000_100.pkl'
	
	all_files = glob.glob(os.path.join(data_dir,'*','*.npy'))
	dict_classes = sort_files(all_files)
	files_chosen = [dict_classes[key_curr][0] for key_curr in dict_classes.keys()]
	per_im_to_keep = 10000
	# pca = get_rand_picked_pca(files_chosen,per_im_to_keep,num_comp_to_keep=100)
	
	# pickle.dump(pca,open(out_file,'wb'))
	# pca = pickle.load(open(out_file,'rb'))
	pca = None

	per_im_to_keep = 1000
	train_files = files_chosen
	test_files = [dict_classes[key_curr][1] for key_curr in dict_classes.keys()]

	features_train_test = []
	labels_train_test = []

	for files_chosen in [train_files,test_files]:
		features = []
		labels = []
		for idx_file_curr,file_curr in enumerate(files_chosen):
			features_curr = random_pick_features(file_curr,per_im_to_keep)
			features = features+features_curr
			labels = labels+[idx_file_curr]* len(features_curr)

		features = np.array(features)
		labels = np.array(labels)
		# print labels.shape, np.unique(labels)
		# print features.shape

		norms = np.linalg.norm(features, axis =1, keepdims = True)
		features = features/norms
		if pca is not None:
			features_pca = pca.transform(features)
			print features_pca.shape
			features_train_test.append(features_pca)
		else:
			features_train_test.append(features)
		labels_train_test.append(labels)

	clf = sklearn.svm.LinearSVC(dual=False, penalty='l1')
	param_grid = [{'C': [0.01, 0.1, 1, 10]}]
	# , 'penalty':['l1','l2']}];

	model = sklearn.model_selection.GridSearchCV(clf, param_grid, n_jobs=1, verbose = 2, error_score=0.0, cv = 3, refit=True)
	model.fit(features_train_test[0],labels_train_test[0])


	# svc = sklearn.svm.SVC(verbose = True)
	# model.fit(features_train_test[0], labels_train_test[0])
	predictions = model.predict(features_train_test[1])
	accuracy = np.sum(predictions==labels_train_test[1])/float(labels_train_test[1].shape[0])
	print accuracy
	import IPython
	IPython.embed()

	# for idx,ratio in enumerate(pca.explained_variance_ratio_):
	# 	print idx,ratio








	# all_files.sort()

	# in_file = all_files[0]
	# print in_file
	# print all_files[-1]
	
	# data = np.load(in_file)
	# data = data[1,:,:]
	# min_val = np.min(data)
	# max_val = np.max(data)
	# print max_val
	# vals = np.linspace(min_val,max_val,10)[1:-1]
	# filter_size = 15
	# filter_curr = (1./filter_size**2)*np.ones((filter_size,filter_size))
	# for val_curr in [0.3]:
	# 	out_arr = np.zeros(data.shape)
	# 	out_arr[data>=val_curr]=1.
		 
	# 	out_arr = scipy.ndimage.filters.convolve(out_arr, filter_curr)
	# 	out_arr[out_arr<val_curr]=0
	# 	out_arr[out_arr>=val_curr]=1.
	# 	plt.ion()
	# 	plt.figure()
	# 	plt.title(str(val_curr))
	# 	plt.imshow(out_arr)
	# raw_input()


	# for in_file in all_files:
		# print np.min(np.min(data,1),1),np.max(np.max(data,1),1)
		# out_file = os.path.join(out_dir,os.path.split(in_file)[1][:-4]+'.jpg')
		# save_viz(in_file, out_file)

	
if __name__=='__main__':
	main()