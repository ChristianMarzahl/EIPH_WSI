import sys
import tensorflow as tf
import numpy as np

ckpt_fpath = "model_rcnn_50/train/baseline_600_L0/model.ckpt-0"

# Open TensorFlow ckpt
reader = tf.train.NewCheckpointReader(ckpt_fpath)

print('\nCount the number of parameters in ckpt file(%s)' % ckpt_fpath)
param_map = reader.get_variable_to_shape_map()
total_count = 0
for k, v in param_map.items():
    if 'Momentum' not in k and 'global_step' not in k:
        temp = np.prod(v)
        total_count += temp
        print('%s: %s => %d' % (k, str(v), temp))

print('Total Param Count: %d' % total_count)