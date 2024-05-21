import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import load_model, save_model
import tensorflow.keras.backend as K

import time
import os
import shutil
import scipy


def load_json_model(model_name):
    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model

def one_hot_to_seg(img_lst_one_hot):
    img_lst_new = []
    for one_hot_img in range(len(img_lst_one_hot)):
        img_new = np.copy(img_lst_one_hot[one_hot_img])
        for lbl in range(0,5):
            img_new[:,:,lbl]*=lbl*255/5
        img_lst_new.append(np.sum(img_new,axis=2))
    return img_lst_new

def one_hot_to_seg_single(img):
    print(img.shape)
    # print(img.shape)
    img_new = np.copy(img)
    for lbl in range(0,5):
        img_new[:,:,lbl]*=lbl*255/5
    # print(img_new.shape)
    result = np.sum(img_new,axis=2)
    print("...")
    print(result.shape)
    return result

def preprocess(image_loc, gnd_truth_loc):
    X_test_test = []
    Y_test_test = []
    n_classes = 5

    img = cv2.imread(image_loc, cv2.IMREAD_GRAYSCALE)

    #         print('E:\YKL\Thorlabs VSCAN Labeling\scan30\{}'.format(img_fl))
    resized_img = cv2.equalizeHist(np.clip(cv2.resize(img,(400, 352), interpolation = cv2.INTER_NEAREST),0,255))
    resized_denoised_img = cv2.fastNlMeansDenoising(resized_img,10,7,21)

    X_test_test.append(resized_denoised_img)

    msk = cv2.imread(gnd_truth_loc, cv2.IMREAD_GRAYSCALE)
    resized_msk = np.clip(cv2.resize(msk,(400, 352), interpolation = cv2.INTER_NEAREST),0,4)

    # additional post processing for one hot encoding mask
    resized_msk[resized_msk==0] = 6
    resized_msk_one_hot = np.zeros((resized_msk.shape[0], resized_msk.shape[1], n_classes))
    for i, unique_value in enumerate(np.unique(resized_msk)):
        resized_msk_one_hot[:, :, i][resized_msk == unique_value] = 1

    Y_test_test.append(resized_msk_one_hot)

    X_test_test = np.array(X_test_test)
    X_test_test = X_test_test[:, :, :, np.newaxis]
    X_test_test = X_test_test / 255

    return X_test_test, Y_test_test

def preprocess_for_inference(image_loc):
    X_test_test = []

    img = cv2.imread(image_loc, cv2.IMREAD_GRAYSCALE)

    #         print('E:\YKL\Thorlabs VSCAN Labeling\scan30\{}'.format(img_fl))
    resized_img = cv2.equalizeHist(np.clip(cv2.resize(img,(400, 352), interpolation = cv2.INTER_NEAREST),0,255))
    resized_denoised_img = cv2.fastNlMeansDenoising(resized_img,10,7,21)

    X_test_test.append(resized_denoised_img)

    X_test_test = np.array(X_test_test)
    X_test_test = X_test_test[:, :, :, np.newaxis]
    X_test_test = X_test_test / 255

    return X_test_test


def read_bscan(img_arr):
    raw_file = np.reshape(img_arr,(800,1024))
    raw_file = raw_file[0:400,:]
    raw_file = np.clip(raw_file, 20, 70, out=raw_file)
    raw_file = (raw_file - 20) / 50
    raw_file = raw_file.T

    raw_file_resized = cv2.resize(raw_file, (640,376))
    raw_file_resized = 255 * raw_file_resized

    plt.imshow(raw_file_resized)
    
    return raw_file_resized.astype(np.uint8)




def run_test():
    loaded_model = load_json_model('models\modelP.json')
    loaded_model.load_weights("models\modelW_last.h5")

    X_test_test, Y_test_test = preprocess("VSCAN_0027_190.png","Label_1_VSCAN_0027_190.png")
    print(X_test_test.shape)
    # X_test_test = preprocess_for_inference(img_path)

    yp_test = loaded_model.predict(x=X_test_test, batch_size=1, verbose=1)
    #yp_test = np.round(yp_test,0)    
    # yp_test_disp = one_hot_to_seg(yp_test)
    yp_test_disp = one_hot_to_seg_single(yp_test[0])
    # print(yp_test[0].shape)
    return yp_test_disp


# if __name__ == '__main__':
#     # initialize()

#     directory = "E:\\MSR2\\SAVES"
#     output_directory = os.path.join(directory,"segmented_mat")
#     if os.path.exists(output_directory):
#         shutil.rmtree(output_directory)
#     os.mkdir(os.path.join(directory,"segmented_mat"))


#     for filename in os.listdir(directory):
#         if filename.endswith(".raw"):
#             full_filepath = os.path.join(directory, filename)
#             full_mat_filepath = os.path.join(output_directory, os.path.splitext(filename)[0]+".mat")
#             with open(full_filepath, 'rb') as f:
#                 raw_file = np.fromfile(f, dtype=np.float32) # add this 
#             out = run_test(raw_file)
#             scipy.io.savemat(full_mat_filepath, {'img': out})
#         else:
#             continue

if __name__ == '__main__':
    img_seg = run_test()
    plt.imshow(img_seg)
    plt.show() 


# if __name__ == '__main__':
    
#     loaded_model = load_json_model('models\modelP.json')
#     loaded_model.load_weights("models\modelW_last.h5")
#     print("num of params {}".format(loaded_model.count_params()))

#     t_start = time.time()
#     # X_test_test, Y_test_test = preprocess("VSCAN_0027_190.png","Label_1_VSCAN_0027_190.png")
#     X_test_test = preprocess_for_inference("VSCAN_0027_190.png")

#     yp_test = loaded_model.predict(x=X_test_test, batch_size=1, verbose=1)
#     #yp_test = np.round(yp_test,0)    
#     # yp_test_disp = one_hot_to_seg(yp_test)
#     yp_test_disp = one_hot_to_seg_single(yp_test[0])
#     print(yp_test[0].shape)

#     print("time elapsed: {}".format(time.time()-t_start))

#     img_seg = yp_test_disp
#     print(img_seg.shape)

#     plt.imshow(img_seg)
#     plt.show() 

#     # t_start = time.time()
#     # X_test_test = preprocess_for_inference("E:\YKL\Thorlabs VSCAN Labeling\scan30\VSCAN_0030_150.png")

#     # yp_test = loaded_model.predict(x=X_test_test, batch_size=1, verbose=1)
#     # #yp_test = np.round(yp_test,0)
#     # yp_test_disp = one_hot_to_seg(yp_test)

#     # print("time elapsed: {}".format(time.time()-t_start))
#     # X_test_test = K.constant(X_test_test)
#     # result = loaded_model.__call__(inputs=X_test_test) # tensorflow Tensor object

#     print("time elapsed: {}".format(time.time()-t_start))
