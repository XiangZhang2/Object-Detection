# coding: utf-8
import cv2
import numpy as np
import tensorflow as tf
from model import darknet
from detect_ops import decode
from utils import preprocess_image, postprocess, draw_detection
from config import anchors, class_names


input_size = (416, 416)

cv2.namedWindow("camera")
capture = cv2.VideoCapture(0)            #开启摄像头
success, image = capture.read()

images = tf.placeholder(tf.float32, [1, input_size[0], input_size[1], 3])
detection_feat = darknet(images)

num = 1

while success and cv2.waitKey(1) == -1:
    cv2.imshow('camera', image)
    image_shape = image.shape[:2]
    image_cp = preprocess_image(image, input_size)
    
    feat_sizes = input_size[0] // 32, input_size[1] // 32
    detection_results = decode(detection_feat, feat_sizes, len(class_names), anchors)

    checkpoint_path = "./checkpoint_dir/yolo2_coco.ckpt"
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        bboxes, obj_probs, class_probs = sess.run(detection_results, feed_dict={images: image_cp})

    bboxes, scores, class_inds = postprocess(bboxes, obj_probs, class_probs,
                                             image_shape=image_shape)
    img_detection = draw_detection(image, bboxes, scores, class_inds, class_names)
    cv2.imwrite("./test/"+str(num)+"test.jpg", img_detection)
    success, image = capture.read()

    num += 1

cv2.destroyWindow("camera")
capture.release()


