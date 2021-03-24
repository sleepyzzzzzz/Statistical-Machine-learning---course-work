import os
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import random

from logreg import RegLogisticRegressor
from softmax import softmax_loss_naive, softmax_loss_vectorized
import linear_classifier

rgb_path = os.path.join('train', 'images', 'rgb')
nir_path = os.path.join('train', 'images', 'nir')
bou_path = os.path.join('train', 'boundaries')
mask_path = os.path.join('train', 'masks')
l1_path = os.path.join('train', 'labels', 'cloud_shadow')
l2_path = os.path.join('train', 'labels', 'double_plant')
l3_path = os.path.join('train', 'labels', 'planter_skip')
l4_path = os.path.join('train', 'labels', 'standing_water')
l5_path = os.path.join('train', 'labels', 'waterway')
l6_path = os.path.join('train', 'labels', 'weed_cluster')
files_rgb = os.listdir(rgb_path) 
files_nir = os.listdir(nir_path) 
files_bou = os.listdir(bou_path) 
files_mask = os.listdir(mask_path) 
files_l1 = os.listdir(l1_path) 
files_l2 = os.listdir(l2_path) 
files_l3 = os.listdir(l3_path) 
files_l4 = os.listdir(l4_path) 
files_l5 = os.listdir(l5_path) 
files_l6 = os.listdir(l6_path)

X = []
y = []

rand_idx = random.sample(range(0, len(files_rgb)), 1000)

for i in range(len(files_rgb)):
    print (i)
    img_rgb = mpimg.imread(os.path.join(rgb_path, files_rgb[i]))
    img_nir = mpimg.imread(os.path.join(nir_path, files_nir[i]))
    img_bou = mpimg.imread(os.path.join(bou_path, files_bou[i]))
    img_mask = mpimg.imread(os.path.join(mask_path, files_mask[i]))
    img_l1 = mpimg.imread(os.path.join(l1_path, files_l1[i]))
    img_l2 = mpimg.imread(os.path.join(l2_path, files_l2[i]))
    img_l3 = mpimg.imread(os.path.join(l3_path, files_l3[i]))
    img_l4 = mpimg.imread(os.path.join(l4_path, files_l4[i]))
    img_l5 = mpimg.imread(os.path.join(l5_path, files_l5[i]))
    img_l6 = mpimg.imread(os.path.join(l6_path, files_l6[i]))
    final_img = np.zeros([img_rgb.shape[0],img_rgb.shape[1],img_rgb.shape[2]+1]);
    final_img[:,:,0] = img_rgb[:,:,0] * img_bou * img_mask
    final_img[:,:,1] = img_rgb[:,:,1] * img_bou * img_mask
    final_img[:,:,2] = img_rgb[:,:,2] * img_bou * img_mask
    final_img[:,:,3] = img_nir * img_bou * img_mask

    mod_label1 = np.where(img_l1 == 1, 1, img_l1)
    mod_label2 = np.where(img_l2 == 1, 2, img_l2)
    mod_label3 = np.where(img_l3 == 1, 3, img_l3)
    mod_label4 = np.where(img_l4 == 1, 4, img_l4)
    mod_label5 = np.where(img_l5 == 1, 5, img_l5)
    mod_label6 = np.where(img_l6 == 1, 6, img_l6)
    final_label = mod_label1 + mod_label2 + mod_label3 + mod_label4 + mod_label5 + mod_label6

    if i in rand_idx:
        X.append(final_img)
        y.append(final_label)

X = np.array(X)
y = np.array(y)
X /= 255

X = X.reshape(1000*512*512, 4)
y = y.reshape((1000*512*512, ))

rand_idx1 = random.sample(range(0, X.shape[0]), 500)
X = np.reshape(X, (-1, 4))
y = np.reshape(y, (-1, ))

# XX = np.vstack([np.ones((X.shape[0],)),X.T]).T

# train
# log_reg
# log_reg = RegLogisticRegressor()

# reg = 100
# theta_opt = log_reg.train(XX,y,reg=reg,num_iters=1000,norm=False)

# print('Theta found by fmin_bfgs: %s' %theta_opt)
# print("Final loss = %.4f" %log_reg.loss(theta_opt,XX,y,0.0))

# log_reg.theta = theta_opt
# predy = log_reg.predict(XX)

# accuracy = float(np.sum(y == predy)) / y.shape[0]
# print("Accuracy on the training set = %.4f" %accuracy)

#softmax
print ('-------------------------------softmax---------------------------')
theta = np.random.randn(X.shape[1], 7) * 0.0001
loss, grad = softmax_loss_vectorized(theta, X, y, 0.0)
print ('loss: ', loss)
results = {}
best_val = -1
best_softmax = None
# learning_rates = [1e-7, 5e-7, 1e-6, 5e-6]
# regularization_strengths = [5e4, 1e5, 5e5, 1e8]
learning_rates = [1e-7, 5e-7, 1e-6]
regularization_strengths = [5e4, 1e5, 5e5]
for lr in learning_rates:
    for rs in regularization_strengths:
        soft = linear_classifier.Softmax()
        soft.train(X, y, learning_rate = lr, reg = rs, num_iters = 4000, batch_size = 400, verbose = False)
        train_acc = np.mean(soft.predict(X) == y)
        val_acc = np.mean(soft.predict(X) == y)
        print (lr, rs, ': ', train_acc, val_acc)
        results[(lr,rs)] = (train_acc, val_acc)
        if val_acc > best_val:
            best_val = val_acc
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: %f' % best_val)

y_test_pred = best_softmax.predict(X)
test_accuracy = np.mean(y == y_test_pred)
print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))

# compute confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y,y_test_pred))


# SVM
print ('-------------------------------SVM---------------------------')
from linear_svm import svm_loss_naive
from linear_classifier1 import LinearSVM
theta1 = np.random.randn(X.shape[1], 7) * 0.0001
loss, grad = svm_loss_naive(theta1, X, y, 0.00001)
print('loss: %f' % (loss, ))
print('grad: %s', grad)

svm = LinearSVM()
y_train_pred = svm.predict(X, grad)
print('training accuracy: %f' % (np.mean(y == y_train_pred), ))
learning_rates = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6]
regularization_strengths = [1e4, 5e4, 1e5, 5e5]

results = {}
best_val = -1
best_svm = None
best_C = 0;
best_lr = 0;

for lr in learning_rates:
    svm = LinearSVM()
    for C in regularization_strengths:
        
        svm.train(X, y, learning_rate=lr, reg=C, num_iters=15000, verbose=False)
        train_pred = svm.predict(X, theta1)
        acc_train = np.mean(y == train_pred)
        results[(lr, C)] = acc_train
        if (acc_train > best_val):
            best_lr = lr
            best_C = C
            best_val = acc_train
            best_svm = svm
for lr, reg in sorted(results):
    train_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f' % (lr, reg, train_accuracy))
    
print('best training accuracy achieved during cross-validation: %f' % best_val)




# # test data
# rgb_path_test = os.path.join('test', 'images', 'rgb')
# nir_path_test = os.path.join('test', 'images', 'nir')
# bou_path_test = os.path.join('test', 'boundaries')
# mask_path_test = os.path.join('test', 'masks')
# files_rgb_test = os.listdir(rgb_path_test) 
# files_nir_test = os.listdir(nir_path_test)
# files_bou_test = os.listdir(bou_path_test)
# files_mask_test = os.listdir(mask_path_test)

# X_test = []
# for i in range(len(files_rgb_test)):
#     print (i)
#     img_rgb_tets = mpimg.imread(os.path.join(rgb_path_tets, files_rgb_tets[i]))
#     img_nir_tets = mpimg.imread(os.path.join(nir_path_tets, files_nir_tets[i]))
#     img_bou_tets = mpimg.imread(os.path.join(bou_path_tets, files_bou_tets[i]))
#     img_mask_tets = mpimg.imread(os.path.join(mask_path_tets, files_mask_tets[i]))
#     final_img_tets = np.zeros([img_rgb_tets.shape[0],img_rgb_tets.shape[1],img_rgb_tets.shape[2]+1]);
#     final_img_tets[:,:,0] = img_rgb_tets[:,:,0] * img_bou_tets * img_mask_tets
#     final_img_tets[:,:,1] = img_rgb_tets[:,:,1] * img_bou_tets * img_mask_tets
#     final_img_tets[:,:,2] = img_rgb_tets[:,:,2] * img_bou_tets * img_mask_tets
#     final_img_tets[:,:,3] = img_nir_tets * img_bou_tets * img_mask_tets

# X_test = np.array(X_test)
# X_test /= 255

# X_mod_test = X_test.reshape(len(files_rgb_test)*512*512, 4)