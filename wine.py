import csv
import tensorflow as tf

def Normalization(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

with open('wine.csv') as f:
    reader = csv.reader(f)
    normal_data = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    x_data = []
    y_data = []
    for row in reader:
        for i in range(0,14):
            normal_data[i].append(float(row[i]))
    for i in range(1,14):
        normal_data[i]=Normalization(normal_data[i])
    length = len(normal_data[0])
    for i in range(0,length):
        ans = [0,0,0]
        ans[int(normal_data[0][i])-1]=1
        y_data.append(ans)
        x_data.append([])
        for j in range(1,14):
            x_data[i].append(normal_data[j][i])

    W = tf.Variable(tf.zeros([13,3]))
    b = tf.Variable(tf.zeros([3]))
    y = tf.matmul(x_data, W) + b
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for step in range(0, 1001):
        sess.run(train_step)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_data, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy))
