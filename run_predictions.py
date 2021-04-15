import os
import numpy as np
import json
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw
from numba import jit 

def check_for_black(I_HSV):
    H = I_HSV[:,:,0]
    S = I_HSV[:,:,1]
    V = I_HSV[:,:,2]

    (n_rows,n_cols) = np.shape(H)
    count = 0 

    for i in range(n_rows):
        for j in range(n_cols):
            if V[i,j] < 45: count += 1
    
    return count > 0.2*n_rows*n_cols

def check_for_red(I_HSV):
    H = I_HSV[:,:,0]
    S = I_HSV[:,:,1]
    V = I_HSV[:,:,2]

    (n_rows,n_cols) = np.shape(H)
    count = 0 

    for i in range(n_rows):
        for j in range(n_cols):
            if (H[i,j] < 40 or H[i,j] > 240) and (S[i,j] > 150) and (V[i,j] > 150): count += 1
    
    return count > 0.05*n_rows*n_cols

def mask(I_HSV):
    H = I_HSV[:,:,0]
    S = I_HSV[:,:,1]
    V = I_HSV[:,:,2]

    (n_rows,n_cols) = np.shape(H)
    I_mask = np.zeros([n_rows,n_cols])

    for i in range(n_rows):
        for j in range(n_cols):
            if (H[i,j] < 40 or H[i,j] > 240) and (S[i,j] > 150) and (V[i,j] > 150): 
                I_mask[i,j] = 1
            else:
                I_mask[i,j] = 0
    
    return I_mask

@jit
def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)
    heatmap = np.zeros((n_rows, n_cols))

    (T_rows,T_cols,T_channels) = np.shape(T)

    row_pad = (T_rows-1)//2 
    col_pad = (T_cols-1)//2

    I_R = I[:,:,0]
    I_G = I[:,:,1]
    I_B = I[:,:,2]

    R_avg = int(np.mean(I_R))
    G_avg = int(np.mean(I_G))
    B_avg = int(np.mean(I_B))

    # Add padding 
    I_pad = np.zeros((n_rows + 2*row_pad, n_cols + 2*col_pad, 3))
   
    I_pad[:,:,0] += R_avg
    I_pad[:,:,1] += G_avg
    I_pad[:,:,2] += B_avg

    I_pad[row_pad:n_rows+row_pad,col_pad:n_cols+col_pad,0] = I_R
    I_pad[row_pad:n_rows+row_pad,col_pad:n_cols+col_pad,1] = I_G
    I_pad[row_pad:n_rows+row_pad,col_pad:n_cols+col_pad,2] = I_B

    (n_rows,n_cols,n_channels) = np.shape(I_pad)

    T = T.flatten()
    T = T.astype(np.float32)
    T = T/(np.sqrt(np.sum(np.square(T))))

    for i in range(n_rows - T_rows):
        for j in range(n_cols - T_cols):

            patch = I_pad[i:i+T_rows,j:j+T_cols,]
            patch = patch.flatten()
            patch = patch.astype(np.float32)
            patch = patch/(np.sqrt(np.sum(np.square(patch))))

            heatmap[i,j] = np.dot(patch, T)
 
    return heatmap

def dist(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def predict_boxes(heatmap, heatmap_mask, I_HSV):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''
    output = []
    heatmap_mask = edge_detection(heatmap_mask)
    clusters = clustering(heatmap_mask)
    
    filtered_clusters = []
    distances = []
    centers = []

    for cluster in clusters:
        length = len(cluster)
        center = sum(i[0]/length for i in cluster), sum(i[1]/length for i in cluster)
        distance = []

        for index in cluster:
            distance.append(dist(index, center))

        avg_distance = sum(distance)/len(distance)

        if np.std(distance)/avg_distance < 1: 
            filtered_clusters.append(cluster)
            distances.append(distance)
            centers.append(center)


    # Predict bounding boxes
    for i in range(len(filtered_clusters)):

        avg_distance = sum(distances[i])/len(distances[i])

        tl_row = centers[i][0] - 2*avg_distance
        tl_col = centers[i][1] - 2*avg_distance
        br_row = centers[i][0] + 8*avg_distance
        br_col = centers[i][1] + 2*avg_distance

        HSV_patch = I_HSV[round(tl_row):round(br_row),round(tl_col):round(br_col),]
        if check_for_black(HSV_patch) and check_for_red(HSV_patch):
            score = heatmap[round(centers[i][0]),round(centers[i][1])]
            score = (score - 0.6)/0.8 + 0.5
            output.append([tl_row,tl_col,br_row - 6*avg_distance,br_col, score])

    '''
    END YOUR CODE
    '''

    return output

def edge_detection(heatmap):
    (n_rows,n_cols) = np.shape(heatmap)
    result = np.copy(heatmap)

    for i in range(1,n_rows-1):
        for j in range(1,n_cols-1):
            if heatmap[i-1,j] and  heatmap[i+1,j] and heatmap[i,j-1] and heatmap[i,j+1]:
                result[i,j] = 0

    return result

def clustering(heatmap):
    (n_rows,n_cols) = np.shape(heatmap) 

    visited = []
    clusters = []
    queue = []

    def append_loc(loc):
        visited.append(loc)
        queue.append(loc)

    for i in range(1,n_rows-1):
        for j in range(1,n_cols-1):
            cluster = []
            if heatmap[i,j] == 255 and not (i,j) in visited: 
                append_loc((i,j))
                cluster.append((i,j))

            while (len(queue) > 0):
                (x,y) = queue.pop(0)

                if heatmap[x-1,y] == 255 and not (x-1,y) in visited: 
                    append_loc((x-1,y))
                    cluster.append((x-1,y))

                if heatmap[x+1,y] == 255 and not (x+1,y) in visited: 
                    append_loc((x+1,y))
                    cluster.append((x+1,y))

                if heatmap[x,y-1] == 255 and not (x,y-1) in visited: 
                    append_loc((x,y-1))
                    cluster.append((x,y-1))

                if heatmap[x,y+1] == 255 and not (x,y+1) in visited: 
                    append_loc((x,y+1))
                    cluster.append((x,y+1))

                if heatmap[x-1,y-1] == 255 and not (x-1,y-1) in visited: 
                    append_loc((x-1,y-1))
                    cluster.append((x-1,y-1))

                if heatmap[x+1,y-1] == 255 and not (x+1,y-1) in visited: 
                    append_loc((x+1,y-1))
                    cluster.append((x+1,y-1))

                if heatmap[x-1,y+1] == 255 and not (x-1,y+1) in visited: 
                    append_loc((x-1,y+1))
                    cluster.append((x-1,y+1))

                if heatmap[x+1,y+1] == 255 and not (x+1,y+1) in visited: 
                    append_loc((x+1,y+1))
                    cluster.append((x+1,y+1))

            if len(cluster) > 5 and len(cluster) < 200: 
                clusters.append(cluster)

    return clusters



def detect_red_light_mf(I, I_HSV):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    heatmap_list = []

    file_names = sorted(os.listdir('../data/hw02_kernels')) 
    file_names = [f for f in file_names if '.jpg' in f] 

    for i in range(len(file_names)):

        T = Image.open(os.path.join('../data/hw02_kernels',file_names[i]))
        T = np.array(T)

        heatmap_list.append(compute_convolution(I, T))

    heatmap_list = np.array(heatmap_list)
    heatmap = np.max(heatmap_list,axis=0)
    
    (rows,cols) = np.shape(heatmap)
    heatmap[round(4*rows/5):,:] = 0

    I_mask = mask(I_HSV)

    heatmap_mask = np.copy(heatmap)
    for i in range(rows):
        for j in range(cols):
            if heatmap[i,j] > 0.6:
                heatmap_mask[i,j] = 1
            else:
                heatmap_mask[i,j] = 0

    # img = Image.fromarray(I_mask*255)
    # img.show() 

    heatmap_mask *= 255*I_mask
    heatmap_mask = heatmap_mask.astype(np.uint8)
    # img = Image.fromarray(heatmap_mask)
    # img.show()
    output = predict_boxes(heatmap, heatmap_mask, I_HSV)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
print(len(file_names_train))
for i in range(len(file_names_train)):
    print(i+1)

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))
    #I.show()

    # convert to numpy array:
    Img = np.array(I)
    Img_HSV = I.convert("HSV")
    Img_HSV = np.array(Img_HSV)
    I.show()

    bounding_boxes = detect_red_light_mf(Img, Img_HSV)

    j=0
    while j < len(bounding_boxes)-1: 
        k = j + 1 
        while k < len(bounding_boxes):
            # diff = np.array(bounding_boxes[k])-np.array(bounding_boxes[j])
            # diff = np.abs(diff)
            # close = max(diff) < 30

            check1 = bounding_boxes[k][0] >= bounding_boxes[j][2] or bounding_boxes[j][0] >= bounding_boxes[k][2]
            check2 = bounding_boxes[k][1] >= bounding_boxes[j][3] or bounding_boxes[j][1] >= bounding_boxes[k][3]

            close = not (check1 or check2) 

            if close: 
                if bounding_boxes[j][4] > bounding_boxes[k][4]: 
                    bounding_boxes.remove(bounding_boxes[k])
                else: 
                    bounding_boxes.remove(bounding_boxes[j])
                j -=1
                break;
            k += 1
        j += 1

    preds_train[file_names_train[i]] = bounding_boxes

    # for box in bounding_boxes:
    #    draw = ImageDraw.Draw(I)  
    #    draw.rectangle([box[1],box[0],box[3],box[2]], fill=None, outline=None, width=1)
    # # save_name = "results_2/output_" + file_names[i] 
    # # I.save(save_name, "JPEG", quality=85)
    # I.show()

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):
        print(len(file_names_test))
        print(i+1)
        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        Img = np.array(I)
        Img_HSV = I.convert("HSV")
        Img_HSV = np.array(Img_HSV)

        preds_test[file_names_test[i]] = detect_red_light_mf(Img, Img_HSV)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
