import tensorflow as tf

print(tf.executing_eagerly())


#x = tf.constant('Hello, TensorFlow!')

#tf.print(x)
#print("This is x",x)

a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
b = tf.add(a, 1)
print(b)
print(a)
