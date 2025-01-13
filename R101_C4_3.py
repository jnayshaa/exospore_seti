import os
from detectron2.data.datasets import register_coco_instances
from ray import tune
from exospore.data_handler.data_loader import register_data_sets
from exospore.training import create_configuration, hyper_tune_trainer, hyper_tune
import psutil

# This needs to be the full path
image_path = '/home/njain/training_mount/njain/test_directory/data/cropped_images'

# This needs to be the full path
train_json_filename = '/home/njain/training_mount/njain/test_directory/data/coco_labels_train.json'

# This needs to be the full path
test_json_filename = '/home/njain/training_mount/njain/test_directory/data/coco_labels_test.json'

# Use a simple model for testing
config_file_url = 'COCO-Detection/faster_rcnn_R_101_C4_3x.yaml'

checkpoint_period = 500  # Increased to reduce memory usage

max_num_epochs = 6000

config = {
        'ims_per_batch': tune.choice([2, 4, 8]),
        'base_lr': tune.loguniform(1e-7, 1e-4),  #Increased from version 2
        'warmup_iters': tune.randint(1000, 2000),  # Increased to stabilize training
        'steps': tune.choice([(500,1000),(1000,2000),(2000,4000)]), #removed (100,200),(200,400),
        'gamma': tune.uniform(0.1, 0.9), 
        'momentum': tune.uniform(0.1, 0.9),
}

hyper_tune(config,
        config_file_url,
	image_path, #positional argument
        checkpoint_period, #positonal argument
        train_json_filename=train_json_filename,
        test_json_filename=test_json_filename,
        num_samples=50, 
        max_num_epochs=max_num_epochs,
        cuda_devices=None)
pass