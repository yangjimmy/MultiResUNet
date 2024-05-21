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

def run_test():
    # run one time inference
    loaded_model = load_json_model('models\modelP.json')
    loaded_model.load_weights("models\modelW_last.h5")
    # print("num of params {}".format(loaded_model.count_params()))

    # t_start = time.time()
    X_test_test, Y_test_test = preprocess("E:\YKL\Thorlabs VSCAN Labeling\scan27\VSCAN_0027_190.png","E:\YKL\Thorlabs VSCAN Labeling\scan27\LabelingProject\GroundTruthProject\PixelLabelData\Label_1_VSCAN_0027_190.png")
    yp_test = loaded_model.predict(x=X_test_test, batch_size=1, verbose=1)
    return one_hot_to_seg(yp_test)


if __name__ == '__main__':
    
    loaded_model = load_json_model('models\modelP.json')
    loaded_model.load_weights("models\modelW_last.h5")
    print("num of params {}".format(loaded_model.count_params()))

    t_start = time.time()
    X_test_test, Y_test_test = preprocess("E:\YKL\Thorlabs VSCAN Labeling\scan27\VSCAN_0027_190.png","E:\YKL\Thorlabs VSCAN Labeling\scan27\LabelingProject\GroundTruthProject\PixelLabelData\Label_1_VSCAN_0027_190.png")

    # yp_test = loaded_model.predict(x=X_test_test, batch_size=1, verbose=1)
    # #yp_test = np.round(yp_test,0)    
    # yp_test_disp = one_hot_to_seg(yp_test)

    # print("time elapsed: {}".format(time.time()-t_start))

    # t_start = time.time()
    # X_test_test = preprocess_for_inference("E:\YKL\Thorlabs VSCAN Labeling\scan30\VSCAN_0030_150.png")

    # yp_test = loaded_model.predict(x=X_test_test, batch_size=1, verbose=1)
    # #yp_test = np.round(yp_test,0)
    # yp_test_disp = one_hot_to_seg(yp_test)

    # print("time elapsed: {}".format(time.time()-t_start))
    X_test_test = K.constant(X_test_test)
    result = loaded_model.__call__(inputs=X_test_test) # tensorflow Tensor object

    print("time elapsed: {}".format(time.time()-t_start))
