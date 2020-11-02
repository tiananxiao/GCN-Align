import tensorflow as tf
import numpy as np
import json
import keras
# tf.enable_eager_execution()
message = tf.constant('welcome to the exciting world of deep nueral networks')


v_1 = tf.constant([1, 3, 4, 5])
v_2 = tf.constant([3, 4, 2, 5])
v_add = tf.add(v_1, v_2)



#  每个会话都需要使用 close() 来明确关闭，而 with 格式可以在运行结束时隐式关闭会话。
zero_t = tf.zeros([3,4,2], tf.float32)
t_random = tf.random_normal([2,3], mean=2.0, stddev=4, seed=12)
t_a = tf.Variable(tf.random_normal([100,100], stddev=2))
t_placeholder = tf.placeholder(tf.float32)
random_float  = tf.zeros(shape=(2,3), dtype=tf.float32, name='dfj')
C = tf.add(random_float, t_random)

# 自动求导机制 tf.GradientTape()
# x = tf.Variable(initial_value=3.0)
# X = tf.constant([[1., 2.], [3., 4.]])
# y = tf.constant([[1.], [2.]])
# w = tf.Variable(initial_value=[[1.], [2.]])
# b = tf.Variable(initial_value=1.)
# with tf.GradientTape() as tape:
#     #f = tf.square(x)
#     L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
# w_grad, b_grad = tape.gradient(L, [w,b])
# print(L, w_grad, b_grad)
#f_grad = tape.gradient(f,x)
#print(f, f_grad)
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]
num_epoch = 1000
optimizer = tf.keras.optimizers.SGD(learning_rate = 5e-4)
for e in range(num_epoch):
    with tf.GradientTape as tape:
        y_pred = a*X +b
        loss = tf.reduce_sum(tf.square(y_pred - y))
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(grads_and_vars = zip(grads, variables))
    print(loss)
with tf.Session() as sess:
    y = 2 * t_placeholder
    print(sess.run(message).decode())
    print(sess.run(v_add))
    print(sess.run(zero_t))
    print(sess.run(y, feed_dict={t_placeholder: 5}))
    print(sess.run(C))

# 建立了一个继承了 tf.keras.Model 的模型类 Linear 。这个类在初始化部分实例化了一个 全连接层 （ tf.keras.layers.Dense ）
class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output
