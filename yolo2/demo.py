"""
Demo for yolov2
"""

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

from model import darknet
from detect_ops import decode
from utils import preprocess_image, postprocess, draw_detection
from config import anchors, class_names


def predict():

    input_size = (416, 416)
    tf.app.flags.DEFINE_string('image_file', './images/person.jpg', 'image_path')
    FLAGS = tf.app.flags.FLAGS
    image_file = FLAGS.image_file
    image = cv2.imread(image_file)
    image_shape = image.shape[:2]
    image_cp = preprocess_image(image, input_size)

    images = tf.placeholder(tf.float32, [1, input_size[0], input_size[1], 3])
    detection_feat = darknet(images)
    feat_sizes = input_size[0] // 32, input_size[1] // 32
    detection_results = decode(detection_feat, feat_sizes, len(class_names), anchors)

    checkpoint_path = "./checkpoint_dir/yolo2_coco.ckpt"
    #checkpoint_path = "/Users/xiang/Downloads/DeepLearning_tutorials-master/ObjectDetections/yolo2/checkpoint_dir/yolo2_coco.ckpt"
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        bboxes, obj_probs, class_probs = sess.run(detection_results, feed_dict={images: image_cp})

    bboxes, scores, class_inds = postprocess(bboxes, obj_probs, class_probs,
                                             image_shape=image_shape)
    img_detection = draw_detection(image, bboxes, scores, class_inds, class_names)
    cv2.imwrite("./res/detection.jpg", img_detection)
    cv2.imshow("detection results", img_detection)
    print('*****Click the window,and press any key to close!*****')

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    predict()





































