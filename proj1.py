################################################################################
# CSE 253: Programming Assignment 1
# Winter 2020


from os import listdir
import os, random, copy
from PIL import Image
import numpy as np
from collections import defaultdict
import math
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

align = load_data("C:/Users/brand/Desktop/PA1/aligned")
resized = load_data("C:/Users/brand/Desktop/PA1/resized")

#partition data
dataset1 = []
dataset2 = []
dataset3 = []
dataset4 = []
dataset5 = []
dataset6 = []
dataset7 = []
dataset8 = []
dataset9 = []
dataset10 = []


count = 0

bias = [1] * 60
bias = np.array(bias)


for l in [dataset1, dataset2,dataset3,dataset4,dataset5,dataset6,dataset7,dataset8,dataset9,dataset10]:
    
    for i in ["Convertible","Minivan","Pickup","Sedan"]:
        for j in range(15):
            j = count * 15 + j
            if j > 147:
                pass
            else :
                x = align[0][i][j]
                l.append(x)
                
    count = count + 1


dataset10.append(align[0]['Convertible'][148])    
dataset10.append(align[0]['Pickup'][148])    
dataset10.append(align[0]['Pickup'][149])        
dataset10.append(align[0]['Sedan'][148])    
dataset10.append(align[0]['Sedan'][149]) 

dataset1 = np.array(dataset1)
dataset1 = np.reshape(dataset1,(60,200*300))
dataset1 = np.transpose(dataset1)
dataset2 = np.array(dataset2)
dataset2 = np.reshape(dataset2,(60,200*300)) 
dataset2 = np.transpose(dataset2)      
dataset3 = np.array(dataset3)
dataset3 = np.reshape(dataset3,(60,200*300))  
dataset3 = np.transpose(dataset3)
dataset4 = np.array(dataset4)
dataset4 = np.reshape(dataset4,(60,200*300))  
dataset4 = np.transpose(dataset4)
dataset5 = np.array(dataset5)
dataset5 = np.reshape(dataset5,(60,200*300))  
dataset5 = np.transpose(dataset5)
dataset6 = np.array(dataset6)
dataset6 = np.reshape(dataset6,(60,200*300))  
dataset6 = np.transpose(dataset6)
dataset7 = np.array(dataset7)
dataset7 = np.reshape(dataset7,(60,200*300)) 
dataset7 = np.transpose(dataset7)
dataset8 = np.array(dataset8)
dataset8 = np.reshape(dataset8,(60,200*300))  
dataset8 = np.transpose(dataset8)
dataset9 = np.array(dataset9)
dataset9 = np.reshape(dataset9,(60,200*300))  
dataset9 = np.transpose(dataset9)
dataset10 = np.array(dataset10)
dataset10 = np.reshape(dataset10,(57,200*300))  
dataset10 = np.transpose(dataset10)


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


#create labels as one hot encoding
truth1 = np.array([0]*15 + [1] * 15 + [2] *15 + [3]*15)
onehot1 = np.zeros((60,4))

for i in range(60):
    onehot1[i,truth1[i]] = 1
    
truth2 = np.array([0]*15 + [1] * 15 + [2] *15 + [3]*15)
onehot2 = np.zeros((60,4))

for i in range(60):
    onehot2[i,truth2[i]] = 1

truth3 = np.array([0]*15 + [1] * 15 + [2] *15 + [3]*15)
onehot3 = np.zeros((60,4))

for i in range(60):
    onehot3[i,truth3[i]] = 1

truth4 = np.array([0]*15 + [1] * 15 + [2] *15 + [3]*15)
onehot4 = np.zeros((60,4))

for i in range(60):
    onehot4[i,truth4[i]] = 1


truth5 = np.array([0]*15 + [1] * 15 + [2] *15 + [3]*15)
onehot5 = np.zeros((60,4))

for i in range(60):
    onehot5[i,truth5[i]] = 1
    
truth6 = np.array([0]*15 + [1] * 15 + [2] *15 + [3]*15)
onehot6 = np.zeros((60,4))

for i in range(60):
    onehot6[i,truth6[i]] = 1
    
truth7 = np.array([0]*15 + [1] * 15 + [2] *15 + [3]*15)
onehot7 = np.zeros((60,4))

for i in range(60):
    onehot7[i,truth7[i]] = 1
    
truth8 = np.array([0]*15 + [1] * 15 + [2] *15 + [3]*15)
onehot8 = np.zeros((60,4))

for i in range(60):
    onehot8[i,truth8[i]] = 1
    
truth9 = np.array([0]*15 + [1] * 15 + [2] *15 + [3]*15)
onehot9 = np.zeros((60,4))

for i in range(60):
    onehot9[i,truth9[i]] = 1
    
truth10 = np.array([0]*13 + [1] * 13 + [2] *13 + [3]*13 + [0] + [2] + [2] + [3] + [3])
onehot10 = np.zeros((57,4))

for i in range(57):
    onehot10[i,truth10[i]] = 1
    

def softmax(X, T, M, V, VT, B, BL,  alpha):
    '''
    first perform pca on X dataset and projects test,val and training datasets
    onto n components then runs softmax regression via gradient descent 
    
    Args: 
        X - Training dataset 
        T - Training Class labels 
        M - Epochs
        V - validation set 
        VT - validation class labels
        B - Test Set
        BL - B class labels
        alpha - learning rate
        reduction - number of principal components
        
    Returns:
        list of training error for each epoch
        list of validation error for each epoch
        classifcation accuracy 
        entropy/loss on test data for best model parameters
        '''
    
    # perform pca,set dimensions, initialize error lists and biases
    a,b,c,d = PCA(np.transpose(X), 20)
    a = np.transpose(a)
    
    V = np.transpose(V) - b
    V = np.matmul(V, d) / c
    V = np.transpose(V)
    
    B = np.transpose(B) - b
    B = np.matmul(B, d) / c
    B = np.transpose(B)
    
    dimension,sample = np.shape(a)
    bias = np.ones((1,sample))  
    a = np.vstack((bias,a))
    
    dimension1,sample1 = np.shape(V)
    bias = np.ones((1,sample1))  
    V = np.vstack((bias,V))
    
    dimension2,sample2 = np.shape(B)
    bias = np.ones((1,sample2))  
    B = np.vstack((bias,B))
    
    runs = 0
    classificationpct = 0
    epsilon = .0000006
    length,width = np.shape(T)
    length2,width2 = np.shape(a)
    length3,width3 = np.shape(V)
    length4, width4 = np.shape(B)
    errortrain = []
    errorvalst = []
    
    T = np.transpose(T)
    VT = np.transpose(VT)
    BL = np.transpose(BL)
    
    confusion = np.zeros((width,width))
    weight = np.zeros((length2,width))
    y = np.zeros((width, width2))
    yval = np.zeros((width,width3))
    
    while runs < M:
        #calculate outputs
        y = np.matmul(np.transpose(weight), a)
        for i in range(width):
            for j in range(width2):
                y[i][j] = math.exp(y[i][j])
        
        coltot = y.sum(axis=0)
        
        for i in range(width):
            for j in range(width2):
                y[i][j] = y[i][j] / coltot[j]
        
        #training error
        error = 0
        
        for l in range(width2):
            for k in range(width):
                error += T[k][l]* math.log(y[k][l] + epsilon)
        
        error = error/(-width2*width)
        
        errortrain.append(error)
                          
        #validation step
        
        yval = np.matmul(np.transpose(weight), V)
        for i in range(width):
            for j in range(width3):
                yval[i][j] = math.exp(yval[i][j])
        
        coltotval = yval.sum(axis=0)
        
        for i in range(width):
            for j in range(width3):
                yval[i][j] = yval[i][j]/coltotval[j]
         
        
        errorval = 0
        
        for l in range(width3):
            for k in range(width):
                errorval+= VT[k][l] * math.log(yval[k][l] + epsilon)
        
        errorval = errorval/(-width3*width)
        errorvalst.append(errorval)
        
        #if errorval > min(errorvalst):
         #   runs = M + 1
        
        #gradient descent step
        
        for p in range(length):
            for m in range(width):
                for n in range(length2):
                    weight[n][m] = weight[n][m] + alpha * (T[m][p] - y[m][p]) * a[n][p]
    
      
        runs += 1
    
    
    #classification step
    
    ytest = np.matmul(np.transpose(weight), B)
    
    for i in range(width):
        for j in range(width4):
            ytest[i][j] = math.exp(ytest[i][j])
    
    coltottest = ytest.sum(axis=0)
    
    for i in range(width):
        for j in range(width4):
            ytest[i][j] = ytest[i][j]/coltottest[j]
    
    errortest = 0
        
    for l in range(width4):
        for k in range(width):
            errortest += BL[k][l] * math.log(ytest[k][l] + epsilon)
        
    errortest = errortest/(-width4*width)
    
    success = 0
    failure = 0
    
    for m in range(width4):
        confusion[np.argmax(ytest[:,m])][np.argmax(BL[:,m])] +=1
        
        if np.argmax(ytest[:,m]) == np.argmax(BL[:,m]):
            success+= 1
        
        else:
            failure += 1 

    classificationpct = success / (success + failure)
    
    
    
    return y,errortrain,errorvalst,classificationpct,errortest,confusion

"""
datasetexp = np.concatenate((dataset1,dataset2,dataset3,dataset4,dataset5,dataset6,dataset7,dataset8), axis = 1)
truthexp = np.concatenate((onehot1,onehot2,onehot3,onehot4,onehot5,onehot6,onehot7,onehot8), axis = 0)


e,f,g,h,z,zz = softmax(datasetexp,truthexp,300,dataset9,onehot9, dataset10, onehot10, .01)
"""

lst = [dataset1, dataset2,dataset3,dataset4,dataset5,dataset6,dataset7,dataset8,dataset9,dataset10]
lst2 = [onehot1,onehot2,onehot3,onehot4,onehot5,onehot6,onehot7,onehot8,onehot9,onehot10]


def cross_val(data_lst,labels_lst,epoch,rate,folds):
    '''
    Prepares one fold of data as val, one as test set and rest as training data
    then performs softmax on predetermined number of folds
    Args:
        data_lst: list of partitioned data set as np arrays
        labels_lst: list of class labels as np arrays
        epoch: number of epochs for each fold of cross val
        rate: learning rate
        folds: number of cross validation sets
    
    Returns:
        TBD
    '''
    train_error_lst = []
    validation_error = []
    classification_pct = []
    test_error = []
    confusion_matrix = np.zeros((4,4))
    count = 0
    
    while count < 10:
        validation_set = data_lst[count]
        test_set = data_lst[(count + 1) % folds]
        
        train_set = np.concatenate((data_lst[(count + 2) % folds],data_lst[(count + 3) % folds]), axis = 1)
        
        
        for i in range(len(data_lst)):
            if i not in [count, (count + 1) % folds, (count + 2) % folds, (count + 3) % folds]:
                train_set = np.concatenate((train_set,data_lst[i]), axis=1)
        
        
        label_val = labels_lst[count]
        label_test = labels_lst[(count + 1) % folds]
        
        label_train = np.concatenate((labels_lst[(count + 2) % folds], labels_lst[(count + 3) % folds]), axis = 0)
        
        for i in range(len(labels_lst)):
            if i not in [count, (count + 1) % folds, (count + 2) % folds, (count + 3) % folds]:
                label_train = np.concatenate((label_train, labels_lst[i]))
                
            
        a,b,c,d,e,f = softmax(train_set,label_train,epoch,validation_set,label_val,test_set,label_test,rate)       
        train_error_lst.append(b)
        validation_error.append(c)
        classification_pct.append(d)
        test_error.append(e)
        confusion_matrix = confusion_matrix + f
        
        
        count += 1
    
    
    return train_error_lst, validation_error, classification_pct, test_error,confusion_matrix

           
bb,cc,dd,ee,confusion_final = cross_val(lst,lst2,300,.01,10)

dd = np.array(dd)
ee = np.array(ee) 
confusion_final = confusion_final / 597


avg_train_error = (np.array(bb[0]) + np.array(bb[1]) + np.array(bb[2]) + np.array(bb[3]) + np.array(bb[4]) + np.array(bb[5]) + np.array(bb[6]) + np.array(bb[7]) + np.array(bb[8]) + np.array(bb[9]))/10
avg_val_error = (np.array(cc[0]) + np.array(cc[1]) + np.array(cc[2]) + np.array(cc[3]) + np.array(cc[4]) + np.array(cc[5]) + np.array(cc[6]) + np.array(cc[7]) + np.array(cc[8]) + np.array(cc[9]))/10
classif_pct = np.mean(dd)
avg_test_error = np.mean(ee)





#finds sd and mean of results and plots them
last = np.array([bb[0],bb[1],bb[2],bb[3],bb[4],bb[5],bb[6],bb[7],bb[8],bb[9]])
last2 = np.array([cc[0],cc[1],cc[2],cc[3],cc[4],cc[5],cc[6],cc[7],cc[8],cc[9]])

sd_train = np.std(last,axis=0)
sd_val = np.std(last2,axis=0)

xaxis = np.arange(300)
xaxis2 = [49,99,149,199,249,299]


ysd = [avg_train_error[49],avg_train_error[99], avg_train_error[149],avg_train_error[199],avg_train_error[249], avg_train_error[299]]
ysd2 = [avg_val_error[49],avg_val_error[99], avg_val_error[149],avg_val_error[199],avg_val_error[249], avg_val_error[299]]

yerror = [sd_train[49], sd_train[99], sd_train[149], sd_train[199], sd_train[249], sd_train[299]]
yerror2 = [sd_val[49],sd_val[99],sd_val[149],sd_val[199],sd_val[249],sd_val[299]]

plt.plot(xaxis, avg_train_error, 'k--', label='Avg training error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Softmax Regression Loss Using 10 PCs and Alpha = .01')

plt.plot(xaxis, avg_val_error, 'b', label ='Avg validation error')

plt.errorbar(xaxis2,ysd,yerr=yerror, fmt = ' ')
plt.errorbar(xaxis2,ysd2,yerr=yerror2, fmt = ' ')

plt.legend()
plt.savefig('plot1.png')
plt.show()


def softmax_stochastic(X, T, M, V, VT, B, BL,  alpha):
    '''
    first perform pca on X dataset and projects test,val and training datasets
    onto n components then runs softmax regression via stochastic gradient descent 
    
    Args: 
        X - Training dataset 
        T - Training Class labels 
        M - Epochs
        V - validation set 
        VT - validation class labels
        B - Test Set
        BL - B class labels
        alpha - learning rate
        reduction - number of principal components
        
    Returns:
        list of training error for each epoch
        list of validation error for each epoch
        classifcation accuracy 
        entropy/loss on test data for best model parameters
        '''
    
    # perform pca,set dimensions, initialize error lists and biases
    a,b,c,d = PCA(np.transpose(X), 20)
    a = np.transpose(a)
    
    V = np.transpose(V) - b
    V = np.matmul(V, d) / c
    V = np.transpose(V)
    
    B = np.transpose(B) - b
    B = np.matmul(B, d) / c
    B = np.transpose(B)
    
    dimension,sample = np.shape(a)
    bias = np.ones((1,sample))  
    a = np.vstack((bias,a))
    
    dimension1,sample1 = np.shape(V)
    bias = np.ones((1,sample1))  
    V = np.vstack((bias,V))
    
    dimension2,sample2 = np.shape(B)
    bias = np.ones((1,sample2))  
    B = np.vstack((bias,B))
    
    runs = 0
    classificationpct = 0
    epsilon = .0000006
    length,width = np.shape(T)
    length2,width2 = np.shape(a)
    length3,width3 = np.shape(V)
    length4, width4 = np.shape(B)
    errortrain = []
    errorvalst = []
    
    T = np.transpose(T)
    VT = np.transpose(VT)
    BL = np.transpose(BL)
    
    confusion = np.zeros((width,width))
    weight = np.zeros((length2,width))
    y = np.zeros((width, width2))
    yval = np.zeros((width,width3))
    
    stochastic = np.arange(width2)
    
    while runs < M:
        np.random.shuffle(stochastic)
        #calculate outputs
        for i in range(len(stochastic)):
            b = a[:,stochastic[i]]
        
            yt = np.matmul(np.transpose(weight), b)
            for l in range(width):
                yt[l] = math.exp(yt[l])
        
            coltot = yt.sum(axis=0)
        
            for l in range(width):
                yt[l] = yt[l] / coltot
            
            #gradient descent step
            
            for m in range(width):
                for n in range(length2):
                    weight[n][m] += alpha * (T[m][stochastic[i]] - yt[m]) * b[n]
        
        #training error
        error = 0
        
        y = np.matmul(np.transpose(weight), a)
        for i in range(width):
            for j in range(width2):
                y[i][j] = math.exp(y[i][j])
        
        coltot = y.sum(axis=0)
        
        for i in range(width):
            for j in range(width2):
                y[i][j] = y[i][j] / coltot[j]
        
        for l in range(width2):
            for k in range(width):
                error += T[k][l]* math.log(y[k][l] + epsilon)
        
        error = error/(-width2*width)
        
        errortrain.append(error)
                          
        #validation step
        
        yval = np.matmul(np.transpose(weight), V)
        for i in range(width):
            for j in range(width3):
                yval[i][j] = math.exp(yval[i][j])
        
        coltotval = yval.sum(axis=0)
        
        for i in range(width):
            for j in range(width3):
                yval[i][j] = yval[i][j]/coltotval[j]
         
        
        errorval = 0
        
        for l in range(width3):
            for k in range(width):
                errorval+= VT[k][l] * math.log(yval[k][l] + epsilon)
        
        errorval = errorval/(-width3*width)
        errorvalst.append(errorval)
        
      
        runs += 1
    
    
    #classification step
    
    ytest = np.matmul(np.transpose(weight), B)
    
    for i in range(width):
        for j in range(width4):
            ytest[i][j] = math.exp(ytest[i][j])
    
    coltottest = ytest.sum(axis=0)
    
    for i in range(width):
        for j in range(width4):
            ytest[i][j] = ytest[i][j]/coltottest[j]
    
    errortest = 0
        
    for l in range(width4):
        for k in range(width):
            errortest += BL[k][l] * math.log(ytest[k][l] + epsilon)
        
    errortest = errortest/(-width4*width)
    
    success = 0
    failure = 0
    
    for m in range(width4):
        confusion[np.argmax(ytest[:,m])][np.argmax(BL[:,m])] +=1
        
        if np.argmax(ytest[:,m]) == np.argmax(BL[:,m]):
            success+= 1
        
        else:
            failure += 1 

    classificationpct = success / (success + failure)
    
    
    
    return y,errortrain,errorvalst,classificationpct,errortest,confusion


datasetexp = np.concatenate((dataset1,dataset2,dataset3,dataset4,dataset5,dataset6,dataset7,dataset8), axis = 1)
truthexp = np.concatenate((onehot1,onehot2,onehot3,onehot4,onehot5,onehot6,onehot7,onehot8), axis = 0)


aaa,bbb,ccc,ddd,eee,fff = softmax_stochastic(datasetexp,truthexp,300,dataset9,onehot9, dataset10, onehot10, .01)



def cross_val_stochastic(data_lst,labels_lst,epoch,rate,folds):
    '''
    same as cross_val but using stochastic softmax inside function
    Args:
        data_lst: list of partitioned data set as np arrays
        labels_lst: list of class labels as np arrays
        epoch: number of epochs for each fold of cross val
        rate: learning rate
        folds: number of cross validation sets
    
    Returns:
        TBD
    '''
    train_error_lst = []
    validation_error = []
    classification_pct = []
    test_error = []
    confusion_matrix = np.zeros((4,4))
    count = 0
    
    while count < 10:
        validation_set = data_lst[count]
        test_set = data_lst[(count + 1) % folds]
        
        train_set = np.concatenate((data_lst[(count + 2) % folds],data_lst[(count + 3) % folds]), axis = 1)
        
        
        for i in range(len(data_lst)):
            if i not in [count, (count + 1) % folds, (count + 2) % folds, (count + 3) % folds]:
                train_set = np.concatenate((train_set,data_lst[i]), axis=1)
        
        
        label_val = labels_lst[count]
        label_test = labels_lst[(count + 1) % folds]
        
        label_train = np.concatenate((labels_lst[(count + 2) % folds], labels_lst[(count + 3) % folds]), axis = 0)
        
        for i in range(len(labels_lst)):
            if i not in [count, (count + 1) % folds, (count + 2) % folds, (count + 3) % folds]:
                label_train = np.concatenate((label_train, labels_lst[i]))
                
            
        a,b,c,d,e,f = softmax_stochastic(train_set,label_train,epoch,validation_set,label_val,test_set,label_test,rate)       
        train_error_lst.append(b)
        validation_error.append(c)
        classification_pct.append(d)
        test_error.append(e)
        confusion_matrix = confusion_matrix + f
        
        count += 1
    
    
    return train_error_lst, validation_error, classification_pct, test_error,confusion_matrix

bbb,ccc,ddd,eee,confusion_final_stoch = cross_val_stochastic(lst,lst2,300,.01,10)

avg_train_error_stoch = (np.array(bbb[0]) + np.array(bbb[1]) + np.array(bbb[2]) + np.array(bbb[3]) + np.array(bbb[4]) + np.array(bbb[5]) + np.array(bbb[6]) + np.array(bbb[7]) + np.array(bbb[8]) + np.array(bbb[9]))/10
xaxis = np.arange(300)

plt.plot(xaxis, avg_train_error_stoch, 'r', label='Stochastic')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Stochastic Gradient Descent Vs Batch Gradient Descent ')

plt.plot(xaxis, avg_train_error, 'b', label ='Batch')

plt.legend()
plt.savefig('gradientvbatch3.png')
plt.show()

