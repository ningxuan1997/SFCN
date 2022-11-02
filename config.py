
from easydict import EasyDict as edict
import numpy as np
__C = edict()
cfg = __C


# Base options
__C.NAME = 'market'  # model name
__C.GPU_ID = [0]
__C.NUM_CLASS = 576   # Market:751; Duke:702; VIPeR:316; CUHK01:871/486; CUHK03:1367/767
__C.ROOT = '/media/yun/work/nx/code/MuDeep_v2-master_6' # path to your project folder   yanz wangluo youmeiyou wenti

# Train options
__C.TRAIN = edict()
__C.TRAIN.SIZE = (384,384)  # 384 x 128
__C.TRAIN.ROOT = '/media/yun/work/nx/VeRi/image_train'
__C.TRAIN.LR = 0.03
__C.TRAIN.STEPSIZE = 80
__C.TRAIN.MAX_EPOCH = 200
__C.TRAIN.GAMMA = 0.1
__C.TRAIN.SNAPSHOT = 10# step for evaluating
__C.TRAIN.CHECKPOINT = 10  # step for saving model
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.NUM_WORKERS = 8
# __C.TRAIN.NUM_FEATURES = 2048

__C.TRAIN.P = 12  # for TripHard Loss (# id)
__C.TRAIN.K = 4  # for TripHard Loss (# images/id)
__C.TRAIN.MARGIN = 1

# Test optionskep
__C.TEST = edict()
__C.TEST.QUERY = '/media/yun/work/nx/VeRi/image_query'
__C.TEST.GALLERY = '/media/yun/work/nx/VeRi/image_test'
__C.TEST.BATCH_SIZE = 128
