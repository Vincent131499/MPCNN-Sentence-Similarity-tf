# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     sim_compu_utils
   Description :  句子相似度得分计算工具
   Author :       Stephen
   date：          2018/9/3
-------------------------------------------------
   Change Activity:
                   2018/9/3:
-------------------------------------------------
"""
__author__ = 'Stephen'

import tensorflow as tf

def compute_l1_distance(x, y):
    """L1距离：|x1-y1| + |x2-y2| + ...... + |xn-yn|"""
    with tf.name_scope('L1_distance'):
        # tf.reduce_sum：压缩求和，降维，其中axis=1:按行求和，axis=0:按列求和
        d = tf.reduce_sum(tf.abs(tf.subtract(x, y)), axis=1)
        return d

def compute_euclidean_distance(x, y):
    """Euclidean 距离计算：sqrt((x1-y1)^2 + (x2-y2)^2 + ...... + (xn-yn)^2)，其中sqrt为平方根"""
    with tf.name_scope('euclidean_distance'):
        d = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1))
        return d

def compute_cosine_distance(x, y):
    """经典余弦角度计算"""
    with tf.name_scope('cosine_distance'):
        up = tf.reduce_sum(tf.multiply(x, y), axis=1)
        low1 = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
        low2 = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1))
        d = tf.divide(up, tf.multiply(low1, low2))
        return d

# def comU1(x, y):
#     result = [compute_cosine_dinstance(x, y), compute_euclidean_distance(x, y), compute_l1_distance(x, y)]
#     return tf.stack(result, axis=1)
#
# def comU2(x, y):
#     result = [compute_cosine_dinstance(x, y), compute_euclidean_distance(x, y)]
#     return tf.stack(result, axis=1)

def comU1(x, y):
    result = [compute_cosine_distance(x, y), compute_l1_distance(x, y)]
    # result = [compute_euclidean_distance(x, y), compute_euclidean_distance(x, y), compute_euclidean_distance(x, y)]
    return tf.stack(result, axis=1)


def comU2(x, y):
    # result = [compute_cosine_distance(x, y), compute_euclidean_distance(x, y)]
    # return tf.stack(result, axis=1)
    return tf.expand_dims(compute_cosine_distance(x, y), -1)
