from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

class FLAGS:
    dataset = '../voc2012_val.tfrecord'
    val_dataset= None
    tiny = False
    weights = '../checkpoint/yolov3.tf'
    classes = './data/voc2012.names'
    mode = 'eager_tf'
    transfer = 'none'
    size  = 416
    epochs =  2
    batch_size =  8
    learning_rate= 1e-3
    num_classes= 20
    weights_num_classes=None
    yolo_max_boxes = 20

anchors = yolo_anchors
anchor_masks = yolo_anchor_masks

if FLAGS.dataset:
    train_dataset = dataset.load_tfrecord_dataset(FLAGS.dataset, FLAGS.classes, FLAGS.size)
else:
    train_dataset = dataset.load_fake_dataset()
train_dataset = train_dataset.shuffle(buffer_size=512)
train_dataset = train_dataset.batch(FLAGS.batch_size)
train_dataset = train_dataset.map(lambda x, y: (
    dataset.transform_images(x, FLAGS.size),
    dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
train_dataset = train_dataset.prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)

train_ds = next(iter(train_dataset))
print(len(train_ds), type(train_ds), type(train_ds[0]), train_ds[0].shape, type(train_ds[1]), len(train_ds[1]))
print(train_ds[1][0].shape)