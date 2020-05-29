
#MODEL EVALUATION
from skimage.filters import threshold_otsu
from scipy.ndimage import label
from keras.models import load_model
import numpy as np
import os
import glob
import random
import math
import cv2
model_folder= 'denoised_aug_lr=0.00002_attempt'
train_col= np.load('train_splitted.npy')
train_masks= np.load('train_masks_splitted.npy')
test_col= np.load('test_splitted.npy')
test_masks= np.load('test_masks_splitted.npy')
total_col= np.concatenate((train_col,test_col), axis=0)
total_mask=np.concatenate((train_masks,test_masks), axis=0)

val_test_col= total_col[620:]
val_test_mask= total_mask[620:]

val_col= val_test_col[:35]
val_mask= val_test_mask[:35]

test_col= val_test_col[35:]
test_mask= val_test_mask[35:]
denoised_test_col= [cv2.GaussianBlur(x, (5, 5), 0) for x in test_col]
denoised_test_col= np.array(denoised_test_col)

def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn
def model_evaluation(model_folder,test_rgb,test_masks):
    
    
    #results=[]
    wd= os.getcwd()
    path= os.path.join(wd,model_folder)
    files= glob.glob(path + '\\*.h5')
    test_rgb = (test_rgb - 127.5) / 127.5
    #test_masks = (test_rgb - 127.5) / 127.5
    #test_masks = test_masks.reshape(256,256)
    final_results={}
    
    #test_masks[test_masks > 0] =1
    for models in files:
        model= load_model(models)
        predictions= model.predict(test_rgb)
        #predictions= predictions.reshape((256,256))
        mean_prec=[]
        for mask, pred in zip(test_masks,predictions):
            predictions2= (pred+1)/2
            mask= (mask -127.5)/127.5
            mask= (mask+1)/2
            
            thresh = threshold_otsu(predictions2.reshape((256,256)))
            thresh2= threshold_otsu(mask.reshape((256,256)))
            binary = predictions2.reshape((256,256)) > thresh
            binary2= mask.reshape((256,256)) > thresh2
            labeled_array, num_features = label(binary2)
            labeled_array2, num_features2 = label(binary)
            intersection = np.histogram2d(labeled_array.flatten(), labeled_array2.flatten(), bins=(num_features, num_features2))[0]

            area_true = np.histogram(labeled_array, bins = num_features)[0]
            area_pred = np.histogram(labeled_array2, bins = num_features2)[0]
            area_true = np.expand_dims(area_true, -1)
            area_pred = np.expand_dims(area_pred, 0)

            union = area_true + area_pred - intersection

            # Exclude background from the analysis
            intersection = intersection[1:,1:]
            union = union[1:,1:]
            union[union == 0] = 1e-9
            
            iou = intersection / union
            prec = []
            print("Thresh\tTP\tFP\tFN\tPrec.")
            #for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(0.7, iou)
            p = tp / (tp + fp + fn)
            if math.isnan(p):
                print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(0.7, tp, fp, fn, p))
                prec.append(0.0)
                print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
                print('\n')
            else:
                print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(0.7, tp, fp, fn, p))
                prec.append(p)
                print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
                print('\n')
                    
            mean_prec.append(np.mean(prec))
            
            final_results.update({models: np.mean(mean_prec)})
    
    return final_results
    #return final_results
results= model_evaluation(model_folder,denoised_test_col,test_mask)
