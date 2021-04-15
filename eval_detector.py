import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''

    check1 = box_1[0] >= box_2[2] or box_2[0] >= box_1[2]
    check2 = box_1[1] >= box_2[3] or box_2[1] >= box_1[3]

    if check1 or check2: 
        return 0
    else: 
        height = max(box_1[0],box_2[0]) - min(box_1[2],box_2[2])
        width = max(box_1[1],box_2[1]) - min(box_1[3],box_2[3])

        intersection = height*width 

        box_1_area = (box_1[2]-box_1[0]) * (box_1[3]-box_1[1])
        box_2_area = (box_2[2]-box_2[0]) * (box_2[3]-box_2[1])

        union = box_1_area + box_2_area - intersection
    
    iou = intersection/union

    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    GT = 0 
    PR = 0 

    '''
    BEGIN YOUR CODE
    '''

    assigned = []
    filtered_preds = {}

    for pred_file, pred in preds.items():
        filtered_preds[pred_file] = []
        for j in range(len(pred)):
            if pred[j][4] >= conf_thr: 
                filtered_preds[pred_file].append(pred[j])

    for pred_file, pred in filtered_preds.items():
        gt = gts[pred_file]
        GT += len(gt)
        PR += len(pred)
        for i in range(len(gt)):
            matches = []
            ious = []
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])

                if iou > iou_thr and pred[j][:4] not in assigned:
                    matches.append(pred[j][:4])
                    ious.append(iou)

            if len(ious) > 0:
                max_index = np.argmax(ious)
                max_match = matches[max_index]
                assigned.append(max_match)
                TP += 1 

    '''
    END YOUR CODE
    '''

    FN = GT - TP 
    FP = PR - TP 
    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train_weakened.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 

confidence_thrs = []
for fname in preds_train:
    for pred in preds_train[fname]:
        confidence_thrs.append(pred[4])

confidence_thrs = np.sort(np.array(confidence_thrs,dtype=float))
#confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train],dtype=float)) # using (ascending) list of confidence scores as thresholds
tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))

for i, conf_thr in enumerate(confidence_thrs):
    tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.25, conf_thr=conf_thr)
precision = tp_train/(tp_train + fp_train)
recall = tp_train/(tp_train + fn_train) 

plt.figure()
plt.plot(recall, precision)
plt.title("Training Precision vs. Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")

for i, conf_thr in enumerate(confidence_thrs):
    tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)
precision = tp_train/(tp_train + fp_train)
recall = tp_train/(tp_train + fn_train) 
plt.plot(recall, precision)

for i, conf_thr in enumerate(confidence_thrs):
    tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.75, conf_thr=conf_thr)
precision = tp_train/(tp_train + fp_train)
recall = tp_train/(tp_train + fn_train) 
plt.plot(recall, precision)

plt.legend(["iou_thr=0.25","iou_thr=0.5","iou_thr=0.75"])
# Plot training set PR curves

if done_tweaking:
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))

    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_test, gts_test, iou_thr=0.25, conf_thr=conf_thr)
    precision = tp_train/(tp_train + fp_train)
    recall = tp_train/(tp_train + fn_train) 
    plt.figure()
    plt.plot(recall, precision)
    plt.title("Testing Precision vs. Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_test, gts_test, iou_thr=0.5, conf_thr=conf_thr)
    precision = tp_train/(tp_train + fp_train)
    recall = tp_train/(tp_train + fn_train) 
    plt.plot(recall, precision)

    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_test, gts_test, iou_thr=0.75, conf_thr=conf_thr)
    precision = tp_train/(tp_train + fp_train)
    recall = tp_train/(tp_train + fn_train) 
    plt.plot(recall, precision)

    plt.legend(["iou_thr=0.25","iou_thr=0.5","iou_thr=0.75"])
    plt.show()

