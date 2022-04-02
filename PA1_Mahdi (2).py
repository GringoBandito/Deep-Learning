from os import listdir
import os, random, copy
from PIL import Image
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt 
from matplotlib.pyplot import subplots


def load_data(data_dir="./aligned/"):
	""" Load all PNG images stored in your data directory into a list of NumPy
	arrays.

	Args:
		data_dir: The relative directory path to the CompCar image directory.
	Returns:
		images: A dictionary with keys as car types and a list containing images associated with each key.
		cnt: A dictionary that stores the # of images in each car type
	"""
	images = defaultdict(list)

	# Get the list of car type directory:
	for e in listdir(data_dir):
		# excluding any non-directory files
		if not os.path.isdir(os.path.join(data_dir, e)):
			continue
		# Get the list of image file names
		all_files = listdir(os.path.join(data_dir, e))

		for file in all_files:
			# Load only image files as PIL images and convert to NumPy arrays
			if '.jpg' in file:
				img = Image.open(os.path.join(data_dir, e, file))
				images[e].append(np.array(img))

	print("Car types: {} \n".format(list(images.keys())))

	cnt = defaultdict(int)
	for e in images.keys():
		print("{}: {} # of images".format(e, len(images[e])))
		cnt[e] = len(images[e])
	return images, cnt

def PCA(X, n_components):
	"""
	Args:
		X: has shape Mxd where M is the number of images and d is the dimension of each image
		n_components: The number of components you want to project your image onto. 
	
	Returns:
		projected: projected data of shape M x n_components
		mean_image: mean of all images
		top_sqrt_eigen_values: singular values
		top_eigen_vectors: eigenvectors 
	"""

	mean_image = np.average(X, axis = 0)

	msd = X - mean_image # M x d

	smart_cov_matrix = np.matmul(msd, msd.T)
	eigen_values, smart_eigen_vectors = np.linalg.eig(smart_cov_matrix)

	idx = eigen_values.argsort()[::-1]   
	eigen_values = eigen_values[idx]
	smart_eigen_vectors = smart_eigen_vectors[:,idx]

	eigen_vectors = (np.matmul(msd.T, smart_eigen_vectors)).T # M x d

	row_norm = np.sum(np.abs(eigen_vectors)**2,axis=-1)**(1./2) # M

	normalized_eigen_vectors = eigen_vectors/(row_norm.reshape(-1, 1)) # M x d

	top_eigen_vectors = normalized_eigen_vectors[:n_components].T
	top_sqrt_eigen_values = np.sqrt(eigen_values[:n_components])

	projected = np.matmul(msd, top_eigen_vectors)/top_sqrt_eigen_values

	return projected, mean_image, top_sqrt_eigen_values, top_eigen_vectors            


def sigmoid(x):
    return 1. / (1 + np.exp(-x))



class DataLoader(object): 
    def __init__(self, name, PATH = './', PCA_components = 10, total_folds = 10, classes = None): 
        self.name = name 
        self.n_components = PCA_components 
        self.PATH = PATH 
        self.len = 0 
        self.dim = 0
        self.X = None 
        self.Y = None 
        self.dataset = {'X': None, 'Y': None}
        self.labels = {}
        self.total_folds = total_folds
        self.classes = classes 
        #self.dataset_split = None
        self.train = None
        self.validation = None
        self.test = None
        
        self.projected = None
        self.mean_image = None
        self.top_sqrt_eigen_values = None
        self.top_eigen_vectors = None
                
        if self.name == 'aligned': 
            path = self.PATH + 'aligned/'
            img_aligned, cnt_aligned = load_data(path)
            self.preprocess(img_aligned, cnt_aligned)
        elif self.name == 'resized': 
            path = self.PATH + 'resized/'
            img_resized, cnt_resized = load_data(path)
            self.preprocess(img_resized, cnt_resized)
        else: 
            exit('Error: unrecognized dataset')
            
    def preprocess(self, imges, count): 
        if self.classes is None:
            self.len = sum(count.values())
        else: 
            self.len = 0 
            for i in range(len(self.classes)):
                self.len += count[self.classes[i]]
                
        self.dim = np.prod((imges['Convertible'][10]).shape)
        self.X = np.zeros([self.len, self.dim])
        self.Y = np.zeros([self.len])
        cnt = 0 
        for key in imges.keys(): 
            if self.classes is None:
                self.labels[key] = cnt
                self.labels[cnt] = key
                cnt+= 1 
            elif key in self.classes: 
                self.labels[key] = cnt
                self.labels[cnt] = key
                cnt+= 1

        cnt = 0 
        for key in imges.keys(): 
            if self.classes is None:
                for img in imges[key]: 
                    self.X[cnt] = img.reshape(-1)
                    self.Y[cnt] = self.labels[key]
                    cnt+= 1 
            elif key in self.classes: 
                for img in imges[key]: 
                    self.X[cnt] = img.reshape(-1)
                    self.Y[cnt] = self.labels[key]
                    cnt+= 1 
                
        self.dataset['X'] = self.X
        self.dataset['Y'] = self.Y
        
        #n_components = 10
        proj, mean_image, top_sqrt_eigen_values, top_eigen_vectors = PCA(self.dataset['X'], self.n_components)
        
        self.projected = proj
        self.mean_image = mean_image
        self.top_sqrt_eigen_values = top_sqrt_eigen_values
        self.top_eigen_vectors = top_eigen_vectors
        
        assert (np.std(self.projected, axis=0)*np.sqrt(self.len)).all()     ## Sanity-Check standard deviation = 1 
        assert np.any(np.abs(np.average(self.projected, axis=0)) < 1e-10)      ## Sanity-Check mean ~= 0 
        
        self.dataset['X'] = proj
        
        #shape = self.dataset['X'].shape
        #print(f'self.dataset[X]: {shape}')
        
        projected_images = {}
        new_count = {}
        for key in imges.keys(): 
            if self.classes is None:
                projected_images[key] = []
                new_count[key] = count[key]
            elif key in self.classes: 
                projected_images[key] = []
                new_count[key] = count[key]
    
        start = 0
        end = 0
        for key in imges.keys(): 
            if self.classes is None:
                end += len(imges[key])
                #print(f'start: {start}, end: {end}')
                projected_images[key] = self.dataset['X'][start:end,:]
                #print(f'{key}, #{len(projected_images[key])}')
                start = end
            elif key in self.classes: 
                end += len(imges[key])
                #print(f'start: {start}, end: {end}')
                projected_images[key] = self.dataset['X'][start:end,:]
                #print(f'{key}, #{len(projected_images[key])}')
                start = end
        
        self.cross_validation_split(projected_images, new_count, self.total_folds)
        
    def cross_validation_split(self, dataset, cnt_dataset, total_folds):
        
        num_class = len(dataset.keys())
        num_dataset = sum(cnt_dataset.values())
        fold_size = int(num_dataset/total_folds)
        if num_dataset % total_folds == 0: 
            fold_size_per_class = int(fold_size/num_class)
        else: 
            fold_size_per_class = int(fold_size/num_class)+1
        self.dataset_split = [{'X': [], 'Y': []} for _ in range(self.total_folds)]

        ind_dict = {key: list(np.arange(cnt_dataset[key])) for key in cnt_dataset.keys()}

        count = 0
        for i in range(total_folds): 
            if i < total_folds-1:
                for key in ind_dict.keys(): 
                    rand_inds = set(np.random.choice(ind_dict[key], fold_size_per_class, replace=False))
                    ind_dict[key] = list(set(ind_dict[key]) - rand_inds)
                    rand_inds = list(rand_inds)
                    for el in rand_inds: 
                        self.dataset_split[i]['X'].append(dataset[key][el])
                        self.dataset_split[i]['Y'].append(self.labels[key])
            else: 
                for key in ind_dict.keys(): 
                    for el in ind_dict[key]:
                        self.dataset_split[i]['X'].append(dataset[key][el])
                        self.dataset_split[i]['Y'].append(self.labels[key])

            count += len(self.dataset_split[i]['X'])
            length = len(self.dataset_split[i]['X'])
            print(f'fold {i}, data size: {length}')
        #print(f'total count: {count}')
        
        #return self.dataset_split

    def get_dataset(self, validation_fold, test_fold):
        self.validation = self.dataset_split[validation_fold]
        self.validation['X'] = np.array(self.validation['X'])
        self.validation['Y'] = np.array(self.validation['Y']).reshape([-1,1])
        self.test = self.dataset_split[test_fold]
        self.test['X'] = np.array(self.test['X'])
        self.test['Y'] = np.array(self.test['Y']).reshape([-1,1])
        
        self.train = {'X': [], 'Y': []}
        
        for i in range(self.total_folds):
            if (i != validation_fold) and (i != test_fold): 
                permutation = np.random.permutation(len(self.dataset_split[i]['X']))
                
                for el in permutation: 
                    self.train['X'].append(self.dataset_split[i]['X'][el])
                    self.train['Y'].append(self.dataset_split[i]['Y'][el])
        self.train['X'] = np.array(self.train['X'])
        self.train['Y'] = np.array(self.train['Y']).reshape([-1,1])
        return [self.train, self.validation, self.test]
    
    def plot_dataset(self, num):
        items = np.random.choice(np.arange(self.len), num, replace=False)
        fig, ax = subplots(num,1, figsize= (10,40))
        ax = ax.ravel()
        for i in range(len(items)):
            ax[i].imshow(self.X[items[i]].reshape(200,300))
            ax[i].axis('off')
            label = '{} : {}'.format(self.labels[self.Y[i]], self.Y[i])
            ax[i].set_title(label)
            
    def plot_PCA(self): 
        fig, ax = subplots(2,2, figsize= (10,10))
        ax = ax.ravel()
        for i in range(4):
            ax[i].imshow(self.top_eigen_vectors[:,i].reshape(200,300))
            ax[i].axis('off')
            label = '{}'.format(i)
            ax[i].set_title(label)
        savename = 'PCA_' + self.name + '.png'    
        fig.savefig(savename)
        
    


class LogisticRegression(object):
    def __init__(self, dataloader, n_epochs = 300, n_runs = 10, in_dim = 10, out_dim = 1, lr = 0.1):
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.w = np.float64(np.random.randn(self.out_dim, in_dim) * 0.1)
        self.b = np.float64(np.zeros((self.out_dim, 1)))
        self.dataloader = dataloader
        self.lr = lr
        self.cost = 0 
        self.dw = None
        self.db = None 
        self.Z = None
        self.A = None
        self.n_runs = n_runs
        self.n_epochs = n_epochs
        self.train_loss = np.zeros([self.n_runs, self.n_epochs])
        self.val_loss = np.zeros([self.n_runs, self.n_epochs])
        self.test_loss = np.zeros([self.n_runs, self.n_epochs])
        
        self.train_acc = np.zeros([self.n_runs, self.n_epochs])
        self.val_acc = np.zeros([self.n_runs, self.n_epochs])
        self.test_acc = np.zeros([self.n_runs, self.n_epochs])
        
        self.train = None 
        self.val = None 
        self.test = None 
        
        self.best_val_loss = None 
        self.best_w = None
        self.best_b = None
        
        self.best_test_loss = np.zeros([self.n_runs])
        self.best_test_acc = np.zeros([self.n_runs])
    
    def forward(self, X, Y): 
        m = X.shape[0]
        
        self.Z = np.float64(np.dot(X, self.w.T) + self.b)
        self.A = np.float64(sigmoid(self.Z))
        
        self.cost = - np.sum(Y * np.log(self.A) + (1-Y) * np.log(1-self.A)) / m
        
    def backprob(self, X, Y): 
        m = X.shape[0]
        self.dw = np.float64(np.dot((self.A - Y).T, X) / m)
        self.db = np.float64(np.sum((self.A - Y), axis=0, keepdims=True) / m)
        
    def optimize(self): 
        
        self.w = np.float64(self.w - self.lr * self.dw)
        self.b = np.float64(self.b - self.lr * self.db)
        
    def comp_cost(self, X, Y): 
        m = X.shape[0]
        
        Z = np.dot(X, self.w.T) + self.b
        A = sigmoid(Z)
        
        cost = - np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) / m
        return cost
    
    def zero_grads(self):
        self.Z = None 
        self.A = None 
        self.cost = None 
        self.dw = None 
        self.db = None 
        
    def predict(self, X, Y): 
        m = X.shape[0]
        
        self.Z = np.dot(X, self.w.T) + self.b
        self.A = sigmoid(self.Z)

        self.Y_p = (self.A > 0.5)
        correct = (self.Y_p == Y)
        accuracy = np.sum(correct) / m
        
        return accuracy
    
    def predict_best(self, X, Y): 
        m = X.shape[0]
        
        self.Z = np.dot(X, self.best_w.T) + self.best_b
        self.A = sigmoid(self.Z)

        self.Y_p = (self.A > 0.5)
        correct = (self.Y_p == Y)
        accuracy = np.sum(correct) / m
        
        return accuracy
    
    def comp_cost_best(self, X, Y): 
        m = X.shape[0]
        
        Z = np.dot(X, self.best_w.T) + self.best_b
        A = sigmoid(Z)
        
        cost = - np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) / m
        return cost
        
    def fit(self): 
        
        for run in range(self.n_runs): 
            validation_fold = run 
            test_fold = np.mod((run+1), self.n_runs)
            
            print(f'----- Run # {run}, val_fold: {validation_fold}, test_fold: {test_fold} -----')
            [self.train, self.val, self.test] = self.dataloader.get_dataset(validation_fold,test_fold)
            self.w = np.float64(np.random.randn(self.out_dim, self.in_dim) * 0.1)
            self.b = np.float64(np.zeros((self.out_dim, 1)))
            
            for e in range(self.n_epochs): 
                #print('Hi')
                self.forward(self.train['X'], self.train['Y'])
                self.backprob(self.train['X'], self.train['Y'])
                self.optimize()
                self.train_loss[run, e] = self.comp_cost(self.train['X'], self.train['Y'])
                self.train_acc[run, e] = self.predict(self.train['X'], self.train['Y'])
                
                self.val_loss[run, e] = self.comp_cost(self.val['X'], self.val['Y'])
                self.val_acc[run, e] = self.predict(self.val['X'], self.val['Y'])
                
                if self.best_val_loss is None: 
                    self.best_val_loss = self.val_loss[run, e]
                    self.best_w = self.w
                    self.best_b = self.b 
                elif self.val_loss[run, e] < self.best_val_loss: 
                    self.best_val_loss = self.val_loss[run, e]
                    self.best_w = self.w
                    self.best_b = self.b 
                else: 
                    #self.lr /= 5
                    pass
                    
                self.test_loss[run, e] = self.comp_cost(self.test['X'], self.test['Y'])
                self.test_acc[run, e] = self.predict(self.test['X'], self.test['Y'])
                
                print(f'Epoch {e+1}, train_loss: {self.train_loss[run, e]:.2f}, train_acc: {self.train_acc[run, e]:.2f}, val_loss: {self.val_loss[run, e]:.2f}, val_acc: {self.val_acc[run, e]:.2f}')
                
                self.zero_grads()
                
            self.best_test_acc[run] = self.predict_best(self.test['X'], self.test['Y'])
            self.best_test_loss[run] = self.comp_cost_best(self.test['X'], self.test['Y'])
            
            print(f'Best Test Loss: {self.best_test_loss[run]}, Best Test Acc: {self.best_test_acc[run]}')


############################### Part B

myData = DataLoader(name ='resized', PATH = './', PCA_components = 6, total_folds = 10)
myData.plot_PCA()


logreg_resized = LogisticRegression(myData, in_dim = 6, lr=0.5)
logreg_resized.fit()
np.mean(logreg_resized.best_test_acc)
logreg_resized.best_test_acc


def plot_loss(train_loss, val_loss, name): 
    fig, ax = subplots(1,1, figsize= (5,5))
    xx = np.arange(300)
    ax.plot(xx, train_loss)
    ax.plot(xx, val_loss)
    ax.legend(['Train Loss', 'Holdout Loss'], loc = 'best', prop={'size':18})
    ax.set_xlabel('Epoch', fontdict = {'size': 18})
    ax.set_ylabel('Loss', fontdict = {'size': 18})
    ax.set_title('Loss vs Epoch', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    #ax.set_ylim([0.692, 0.694])
    
    savename = 'loss_' + name + '.png' 
    fig.savefig(savename, bbox_inches = "tight")


plot_loss(logreg_resized.train_loss[0], logreg_resized.val_loss[0], 'p2')


def plot_acc(train_acc, val_acc, name): 
    fig, ax = subplots(1,1, figsize= (5,5))
    xx = np.arange(300)
    ax.plot(xx, train_acc)
    ax.plot(xx, val_acc)
    ax.legend(['Train Accuracy', 'Holdout Accuracy'], loc = 'best', prop={'size':18})
    ax.set_xlabel('Epoch', fontdict = {'size': 18})
    ax.set_ylabel('Accuracy', fontdict = {'size': 18})
    ax.set_title('Accuracy vs Epoch', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    #ax.set_ylim([0.692, 0.694])
    
    savename = 'acc_' + name + '.png' 
    fig.savefig(savename, bbox_inches = "tight")

plot_acc(logreg_resized.train_acc[0], logreg_resized.val_acc[0], 'p2')


################################# PART C 

myData = DataLoader(name ='aligned', PATH = './', PCA_components = 6, total_folds = 10, classes = ['Convertible', 'Minivan'])
logreg_resized = LogisticRegression(myData, in_dim = 6, lr=1)
myData.plot_PCA()
logreg_resized.fit()

print(np.mean(logreg_resized.best_test_acc))

logreg_resized.best_test_acc
print(np.std(logreg_resized.best_test_acc))


def plot_errorbar_loss(train_loss, val_loss, name): 
    fig, ax = subplots(1,1, figsize= (5,5))
    xx = np.arange(300)
    mean_train_loss = np.mean(train_loss, axis=0)
    std_train_loss = np.std(train_loss, axis=0)
    mean_val_loss = np.mean(val_loss, axis=0)
    std_val_loss = np.std(val_loss, axis=0)
    std_train_loss2 = np.zeros([300])
    std_val_loss2 = np.zeros([300])
    for i in range(300): 
        if (i+1)%50 == 0: 
            std_train_loss2[i] = std_train_loss[i]
            std_val_loss2[i] = std_val_loss[i]
    ax.errorbar(xx, mean_train_loss, yerr=std_train_loss2, ecolor = 'red', elinewidth= 8)
    ax.errorbar(xx, mean_val_loss, yerr=std_val_loss2, ecolor = 'green')
    
    ax.legend(['Train Loss', 'Holdout Loss'], loc = 'best', prop={'size':18})
    ax.set_xlabel('Epoch', fontdict = {'size': 18})
    ax.set_ylabel('Loss', fontdict = {'size': 18})
    ax.set_title('Loss vs Epoch', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    #ax.set_ylim([0.692, 0.694])
    
    savename = 'loss_' + name + '.png' 
    fig.savefig(savename, bbox_inches = "tight")

def plot_errorbar_acc(train_acc, val_acc, name): 
    fig, ax = subplots(1,1, figsize= (5,5))
    xx = np.arange(300)
    mean_train_acc = np.mean(train_acc, axis=0)
    std_train_acc = np.std(train_acc, axis=0)
    mean_val_acc = np.mean(val_acc, axis=0)
    std_val_acc = np.std(val_acc, axis=0)
    std_train_acc2 = np.zeros([300])
    std_val_acc2 = np.zeros([300])
    for i in range(300): 
        if (i+1)%50 == 0: 
            std_train_acc2[i] = std_train_acc[i]
            std_val_acc2[i] = std_val_acc[i]
    ax.errorbar(xx, mean_train_acc, yerr=std_train_acc2, ecolor = 'red', elinewidth= 8)
    ax.errorbar(xx, mean_val_acc, yerr=std_val_acc2, ecolor = 'green')
    ax.legend(['Train Accuracy', 'Holdout Accuracy'], loc = 'lower left', prop={'size':18})
    #leg = ax.get_legend()
    #leg.legendHandles[0].set_color('red')
    #leg.legendHandles[1].set_color('yellow')
    ax.set_xlabel('Epoch', fontdict = {'size': 18})
    ax.set_ylabel('Accuracy', fontdict = {'size': 18})
    ax.set_title('Accuracy vs Epoch', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    #ax.set_ylim([0.692, 0.694])
    
    savename = 'acc_' + name + '.png' 
    fig.savefig(savename, bbox_inches = "tight")


plot_errorbar_loss(logreg_resized.train_loss, logreg_resized.val_loss, 'p3_i')
plot_errorbar_acc(logreg_resized.train_acc, logreg_resized.val_acc, 'p3_i')

logreg_resized = LogisticRegression(myData, in_dim = 6, lr=0.1)
logreg_resized.fit()
train_loss1 = logreg_resized.train_loss

logreg_resized = LogisticRegression(myData, in_dim = 6, lr=8)
logreg_resized.fit()
train_loss2 = logreg_resized.train_loss

logreg_resized = LogisticRegression(myData, in_dim = 6, lr=10)
logreg_resized.fit()
train_loss3 = logreg_resized.train_loss


def plot_errorbar_loss123(train_loss1, train_loss2, train_loss3, name): 
    fig, ax = subplots(1,1, figsize= (5,5))
    xx = np.arange(300)
    mean_train_loss1 = np.mean(train_loss1, axis=0)
    std_train_loss1 = np.std(train_loss1, axis=0)
    mean_train_loss2 = np.mean(train_loss2, axis=0)
    std_train_loss2 = np.std(train_loss2, axis=0)
    mean_train_loss3 = np.mean(train_loss3, axis=0)
    std_train_loss3 = np.std(train_loss3, axis=0)
    std_train_loss12 = np.zeros([300])
    std_train_loss22 = np.zeros([300])
    std_train_loss32 = np.zeros([300])
    for i in range(300): 
        if (i+1)%50 == 0: 
            std_train_loss12[i] = std_train_loss1[i]
            std_train_loss22[i] = std_train_loss2[i]
            std_train_loss32[i] = std_train_loss3[i]
    ax.errorbar(xx, mean_train_loss1, yerr=std_train_loss12)
    ax.errorbar(xx, mean_train_loss2, yerr=std_train_loss22)
    ax.errorbar(xx, mean_train_loss3, yerr=std_train_loss32)
    
    ax.legend(['lr: 0.1', 'lr: 8', 'lr: 10'], loc = 'best', prop={'size':18})
    ax.set_xlabel('Epoch', fontdict = {'size': 18})
    ax.set_ylabel('Loss', fontdict = {'size': 18})
    ax.set_title('Loss vs Epoch', fontdict = {'size': 18})
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    #ax.set_ylim([0.692, 0.694])
    
    savename = 'loss_' + name + '.png' 
    fig.savefig(savename, bbox_inches = "tight")


name = 'p3_iii'
plot_errorbar_loss123(train_loss1, train_loss2, train_loss3, name)


################################## PART D 

myData = DataLoader(name ='aligned', PATH = './', PCA_components = 6, total_folds = 10, classes = ['Sedan', 'Pickup'])
logreg_resized = LogisticRegression(myData, in_dim = 6, lr=8)

logreg_resized.fit()

plot_errorbar_loss(logreg_resized.train_loss, logreg_resized.val_loss, 'p4_i')
plot_errorbar_acc(logreg_resized.train_acc, logreg_resized.val_acc, 'p4_i')

print(np.mean(logreg_resized.best_test_acc))
print(np.std(logreg_resized.best_test_acc))

