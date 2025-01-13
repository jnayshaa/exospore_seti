# Tests using Ray Tune to hyper-parameter tune
import os
import time
from detectron2.data.datasets import register_coco_instances
from ray import tune
from exospore.data_handler.data_loader import register_data_sets
from exospore.training import create_configuration, hyper_tune_trainer, hyper_tune
# from exospore.data_handler.sampler import generate_anchor_sizes, generate_anchor_ratios
import psutil

# outerStartTime = time.time()

# This needs to be the full path
image_path = '/home/njain/training_mount/njain/test_directory/data/cropped_images'

# This needs to be the full path
train_json_filename = '/home/njain/training_mount/njain/test_directory/data/coco_labels_train.json'

# This needs to be the full path
test_json_filename = '/home/njain/training_mount/njain/test_directory/data/coco_labels_test.json'

# Model to use
config_file_url = 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'

checkpoint_period = 100
max_num_epochs = 12000

config = {
        'ims_per_batch': 10,
        'base_lr': tune.loguniform(1e-4, 1e-5),
        'warmup_iters': tune.randint(400, 500),
        'steps': (2000,4000),
        'gamma': tune.uniform(0.4, 0.6),
        'momentum': tune.uniform(0.7,0.9),
        }


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