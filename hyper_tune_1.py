# Tests using Ray Tune to hyper-parameter tune

import os

from detectron2.data.datasets import register_coco_instances

from ray import tune

from exospore.data_handler.data_loader import register_data_sets
from exospore.training import create_configuration, hyper_tune_trainer, hyper_tune

# This needs to be the full path
image_path = '/home/njain/training_mount/njain/test_directory/data/cropped_images'

# This needs to be the full path
train_json_filename = '/home/njain/training_mount/njain/test_directory/data/coco_labels_train.json'

# This needs to be the full path
test_json_filename = '/home/njain/training_mount/njain/test_directory/data/coco_labels_test.json'

# Use a simple model for testing
config_file_url = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
# config_file_url = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'

#checkpoint_period = 10
#max_num_epochs = 500
checkpoint_period = 100
max_num_epochs = 6000

#       'base_lr': tune.loguniform(1e-4, 1e-3),
#       'steps': tune.choice([(100,200),(200,400),(500,1000),(1000,2000),(2000,4000)]),
config = {
        'ims_per_batch': tune.choice([8, 16, 32]),
        'base_lr': tune.loguniform(1e-5, 1e-3),
        'warmup_iters': tune.randint(100, 1000),
        'steps': tune.choice([(100,200),(200,400),(500,1000),(1000,2000),(2000,4000)]),
        'gamma': tune.uniform(0.1, 0.9),
        'momentum': tune.uniform(0.1,0.9)
        }

"""
hyper_tune(config,
        config_file_url,
        train_json_filename,
        test_json_filename,
        image_path,
        checkpoint_period,
        num_samples=4, 
        max_num_epochs=max_num_epochs,
        cuda_devices=None)

"""
hyper_tune(config,
        config_file_url,
        train_json_filename,
        test_json_filename,
        image_path,
        checkpoint_period,
        num_samples=50, 
        max_num_epochs=max_num_epochs,
        cuda_devices=None)

pass