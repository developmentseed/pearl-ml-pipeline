import sys
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from sklearn.metrics import f1_score

NAIP_2013_MEANS = np.array([117.00, 130.75, 122.50, 159.30])
NAIP_2013_STDS = np.array([38.16, 36.68, 24.30, 66.22])
NAIP_2017_MEANS = np.array([72.84,  86.83, 76.78, 130.82])
NAIP_2017_STDS = np.array([41.78, 34.66, 28.76, 58.95])
NAIP_NY_2017_MEANS = np.array([95.31, 129.95, 127.77, 184.45])
NAIP_NY_2017_STDS = np.array([40.95, 34.71, 21.07, 51.11])
NAIP_PA_2017_MEANS = np.array([114.38, 140.45, 110.13, 177.38])
NAIP_PA_2017_STDS = np.array([37.401, 34.29, 23.77, 45.98])
NAIP_DE_2013_MEANS = np.array([116.74, 132.48, 127.61, 175.80])
NAIP_DE_2013_STDS = np.array([40.22, 34.22, 23.86, 60.58])
NAIP_VA_2018_MEANS = np.array([92.70, 104.10, 75.43, 118.62])
NAIP_VA_2018_STDS = np.array([42.56, 42.91, 31.27, 59.34])
NAIP_WV_2018_MEANS = np.array([109.36, 123.73, 105.63, 117.47])
NAIP_WV_2018_STDS = np.array([44.79, 36.71, 31.72, 38.93])
NAIP_MD_2018_MEANS = np.array([103.84, 108.31,  88.07, 113.36])
NAIP_MD_2018_STDS = np.array([46.19, 44.21, 37.07, 60.50])
NAIP_MD_2017_MEANS = np.array([73.31, 86.94,  77.38, 126.26])
NAIP_MD_2017_STDS = np.array([42.22, 35.90, 30.23, 60.49])
NAIP_MD_2015_MEANS = np.array([116.05, 126.48,  117.93, 158.21])
NAIP_MD_2015_STDS = np.array([38.08, 32.87, 27.07, 83.22])
NAIP_MD_2011_MEANS = np.array([108.35,  126.13, 121.83, 176.64])
NAIP_MD_2011_STDS = np.array([39.94, 30.60, 25.39, 45.69])
NAIP_MD_Merged_MEANS = np.array([100.3875, 111.965, 101.3025, 143.6175])
NAIP_MD_Merged_STDS = np.array([58.71, 59.7775, 54.05, 95.2125])
NAIP_VA_Merged_MEANS = np.array([102.6326, 118.5233, 104.2282,145.7618])
NAIP_VA_Merged_STDS = np.array([39.6812, 37.2886, 32.8185, 50.1053])
NAIP_NY_Merged_MEANS = np.array([97.2829, 122.9519, 106.3612,  169.0045])
NAIP_NY_Merged_STDS = np.array([39.7267, 36.6849, 25.8357, 52.4304])
fresno_ca_means = np.array([132.70, 127.63, 109.55, 147.25])
fresno_ca_stds = np.array([45.21, 38.478,  34.65, 35.70])
la_ca_means = np.array([115.06,114.08, 105.04, 123.96])
la_ca_stds = np.array([56.31, 49.89, 44.26, 48.78])
sanoma_ca_means = np.array([93.69, 101.96,  90.17, 126.93])
sanoma_ca_stds = np.array([49.12,39.83, 33.27, 54.14])



NLCD_CLASSES = [ 0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95] # 16 classes + 1 nodata class ("0"). Note that "12" is "Perennial Ice/Snow" and is not present in Maryland.

NLCD_CLASS_COLORMAP = { # Copied from the emebedded color table in the NLCD data files
    0:  (0, 0, 0, 255),
    11: (70, 107, 159, 255),
    12: (209, 222, 248, 255),
    21: (222, 197, 197, 255),
    22: (217, 146, 130, 255),
    23: (235, 0, 0, 255),
    24: (171, 0, 0, 255),
    31: (179, 172, 159, 255),
    41: (104, 171, 95, 255),
    42: (28, 95, 44, 255),
    43: (181, 197, 143, 255),
    52: (204, 184, 121, 255),
    71: (223, 223, 194, 255),
    81: (220, 217, 57, 255),
    82: (171, 108, 40, 255),
    90: (184, 217, 235, 255),
    95: (108, 159, 184, 255)
}

NLCD_IDX_COLORMAP = {
    idx: NLCD_CLASS_COLORMAP[c]
    for idx, c in enumerate(NLCD_CLASSES)
}
LC_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
LC_CLASS_COLORMAP = {
    0: (0, 0, 0, 0),
    1: (0, 197, 255, 255),
    2: (0, 168, 132, 255),
    3: (38, 115, 0, 255),
    4: (76, 230, 0, 255),
    5: (163, 255, 115, 255),
    6: (255, 170, 0, 255),
    7: (255, 0, 0, 255),
    8: (156, 156, 156, 255),
    9: (0, 0, 0, 255),
    10: (115, 115, 0, 255),
    11: (230, 230, 0, 255),
    12: (255, 255, 115, 255),
    13: (197, 0, 255, 255)
}

LC_CLASSES_TREE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
LC_CLASS_TREE_COLORMAP = {
    0: (252, 232, 3),
    1: (0, 197, 255, 255),
    2: (0, 168, 132, 255),
    3: (38, 115, 0, 255),
    4: (76, 230, 0, 255),
    5: (163, 255, 115, 255),
    6: (255, 170, 0, 255),
    7: (255, 0, 0, 255),
    8: (156, 156, 156, 255),
    9: (0, 0, 0, 255)
}


LC_COLORMAP = {
    idx: LC_CLASS_COLORMAP[c]
    for idx, c in enumerate(LC_CLASSES)
}


LC_TREE_COLORMAP = {
    idx: LC_CLASS_TREE_COLORMAP[c]
    for idx, c in enumerate(LC_CLASSES_TREE)
}

EPA_CLASSES = [0, 10, 20, 30, 40, 52, 70, 80, 82, 91, 92]

epa_label_dict = {0: 0, 10: 1, 20: 2, 30: 3, 40: 4, 52: 5, 70: 6, 80: 7, 82: 8, 91: 9, 92: 10}

CIC_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8]

cic_label_dict = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}

naip_5cls = {1: 0, 2: 0, 3: 1, 10: 1, 11: 1, 12: 1, 4: 2, 5: 2, 6: 3, 7: 4, 8: 4, 9: 4}

naip_4cls = {1: 0, 2: 0, 3: 1, 10: 1, 11: 1, 12: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3}

uvm_7cls = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6}

uvm_8cls = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}


def get_nlcd_class_to_idx_map():
    nlcd_label_to_idx_map = []
    idx = 0
    for i in range(NLCD_CLASSES[-1]+1):
        if i in NLCD_CLASSES:
            nlcd_label_to_idx_map.append(idx)
            idx += 1
        else:
            nlcd_label_to_idx_map.append(0)
    nlcd_label_to_idx_map = np.array(nlcd_label_to_idx_map).astype(np.int64)
    return nlcd_label_to_idx_map

NLCD_CLASS_TO_IDX_MAP = get_nlcd_class_to_idx_map()  # I do this computation on import for illustration (this could instead be a length 96 vector that is hardcoded here)


NLCD_IDX_TO_REDUCED_LC_MAP = np.array([
    4,#  0 No data 0
    0,#  1 Open Water
    4,#  2 Ice/Snow
    2,#  3 Developed Open Space
    3,#  4 Developed Low Intensity
    3,#  5 Developed Medium Intensity
    3,#  6 Developed High Intensity
    3,#  7 Barren Land
    1,#  8 Deciduous Forest
    1,#  9 Evergreen Forest
    1,# 10 Mixed Forest
    1,# 11 Shrub/Scrub
    2,# 12 Grassland/Herbaceous
    2,# 13 Pasture/Hay
    2,# 14 Cultivated Crops
    1,# 15 Woody Wetlands
    1,# 16 Emergent Herbaceious Wetlands
])

NLCD_IDX_TO_REDUCED_LC_ACCUMULATOR = np.array([
    [0,0,0,0,1],#  0 No data 0
    [1,0,0,0,0],#  1 Open Water
    [0,0,0,0,1],#  2 Ice/Snow
    [0,0,0,0,0],#  3 Developed Open Space
    [0,0,0,0,0],#  4 Developed Low Intensity
    [0,0,0,1,0],#  5 Developed Medium Intensity
    [0,0,0,1,0],#  6 Developed High Intensity
    [0,0,0,0,0],#  7 Barren Land
    [0,1,0,0,0],#  8 Deciduous Forest
    [0,1,0,0,0],#  9 Evergreen Forest
    [0,1,0,0,0],# 10 Mixed Forest
    [0,1,0,0,0],# 11 Shrub/Scrub
    [0,0,1,0,0],# 12 Grassland/Herbaceous
    [0,0,1,0,0],# 13 Pasture/Hay
    [0,0,1,0,0],# 14 Cultivated Crops
    [0,1,0,0,0],# 15 Woody Wetlands
    [0,1,0,0,0],# 16 Emergent Herbaceious Wetlands
])

class Timer():
    '''A wrapper class for printing out what is running and how long it took.
    Use as:
    ```
    with utils.Timer("running stuff"):
        # do stuff
    ```
    This will output:
    ```
    Starting 'running stuff'
    # any output from 'running stuff'
    Finished 'running stuff' in 12.45 seconds
    ```
    '''
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.tic = float(time.time())
        print("Starting '%s'" % (self.message))

    def __exit__(self, type, value, traceback):
        print("Finished '%s' in %0.4f seconds" % (self.message, time.time() - self.tic))


def fit(model, device, data_loader, num_batches, optimizer, criterion, epoch, memo=''):
    model.train()
    losses = []
    tic = time.time()
    for batch_idx, (data, targets) in tqdm(enumerate(data_loader), total=num_batches, file=sys.stdout):
        data = data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        # error Expected more than 1 value per channel when training
        # check https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274
        model.eval()
        outputs = model(data)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    avg_loss = np.mean(losses)
    print('[{}] Training Epoch: {}\t Time elapsed: {:.2f} seconds\t Loss: {:.2f}'.format(
        memo, epoch, time.time()-tic, avg_loss), end=""
    )
    print("")
    return [avg_loss]

def evaluate(model, device, data_loader, num_batches, criterion, epoch, memo=''):
    model.eval()
    losses = []
    tic = time.time()
    for batch_idx, (data, targets) in tqdm(enumerate(data_loader), total=num_batches, file=sys.stdout):
        data = data.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            outputs = model(data)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
    avg_loss = np.mean(losses)

    print('[{}] Validation Epoch: {}\t Time elapsed: {:.2f} seconds\t Loss: {:.2f}'.format(
        memo, epoch, time.time()-tic, avg_loss), end=""
    )
    print("")
    return [avg_loss]

def score(model, device, data_loader, num_batches):
    model.eval()

    num_classes = model.module.segmentation_head[0].out_channels
    num_samples = len(data_loader.dataset)
    predictions = np.zeros((num_samples, num_classes), dtype=np.float32)
    idx = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        with torch.no_grad():
            output = F.softmax(model(data))
        batch_size = data.shape[0]
        predictions[idx:idx+batch_size] = output.cpu().numpy()
        idx += batch_size
    return predictions

def score2(model, device, data_loader, num_batches, num_classes):
    model.eval()

    predictions = []
    ground_truth = []
    idx = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = F.softmax(model(data))
        output = output.cpu().numpy()  # (32, 10, 256, 256)
        target = target.cpu().numpy()

        for i, x in enumerate(output):
            predictions.append(x.argmax(axis=0).astype(np.uint8))
            ground_truth.append(target[i])


    # to this per batch instead of all at once to fix memory errors
    preds_f = np.reshape(np.array(predictions), [-1])
    gt_f = np.reshape(np.array(ground_truth), [-1])
    per_class_f1 = f1_score(gt_f, preds_f, average=None)
    global_f1 = f1_score(gt_f, preds_f, average='weighted')

    return per_class_f1, global_f1

def score_batch(model, device, data_loader, num_batches, num_classes):
    model.eval()

    batch_per_class_f1 = []
    batch_global_f1 = []
    idx = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        predictions = []
        ground_truth = []
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = F.softmax(model(data))
        output = output.cpu().numpy()
        target = target.cpu().numpy()

        print('output')
        print(output)
        for i, x in enumerate(output):
            predictions.append(x.argmax(axis=0).astype(np.uint8))
            ground_truth.append(target[i])

        preds_f = np.reshape(np.array(predictions), [-1])
        print('ground truth')
        print(ground_truth)
        gt_f = np.reshape(np.array(ground_truth), [-1])

        missing_labels = np.setdiff1d(list(np.arange(num_classes)), np.unique(gt_f))


        per_class_f1 = f1_score(gt_f, preds_f, average=None)

        per_class_f1_final = np.zeros(num_classes)
        # add nan for missing label classes
        for x in missing_labels:
            per_class_f1_final[x] = np.nan
        print('gt_f')
        print(gt_f)
        print('np unique')
        print(np.unique(gt_f))
        for i, gt_class in enumerate(np.unique(gt_f)):
            per_class_f1_final[gt_class] = per_class_f1[i]


        global_f1 = f1_score(gt_f, preds_f, average='weighted')

        batch_per_class_f1.append(per_class_f1_final)
        batch_global_f1.append(global_f1)

    batch_per_class_f1_mean = np.nanmean(batch_per_class_f1, axis = 0)
    batch_global_f1_mean = np.mean(batch_global_f1)
    return batch_per_class_f1_mean, batch_global_f1_mean


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
