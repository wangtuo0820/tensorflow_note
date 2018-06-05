import tensorflow as tf
from PIL import Image

filename_queue = tf.train.string_input_producer(['train.tfrecords']) # 生成一个队列
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue) # 返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                    features={
                                        'label':tf.FixedLenFeature([], tf.int64),
                                        'img_raw':tf.FixedLenFeature([], tf.string)}) # 将img数据和label取出来1
img = tf.decode_raw(features['img_raw'], tf.uint8)
img = tf.reshape(img, [128,128,3])
label = tf.cast(features['label'], tf.int32)
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(4):
        example, l = sess.run([img, label])
        image = Image.fromarray(example, 'RGB')
        image.save('./'+str(i)+'_'+str(l)+'.jpg')
        print(example, l)
    coord.request_stop()
    coord.join(threads)
