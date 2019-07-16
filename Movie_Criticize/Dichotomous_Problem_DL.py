from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import Draw_Diagram

# 加载IMDB数据集  num_word表示保留词频为前10000的单词
# data为2维张量,第一维度表示评论,第二维度表示评论内的单词
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# 由于评论数据是个不定长度的整数序列,因此必须将其处理为张量
# 以下为手动实现方法
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        # '1.'表示浮点类型数据
        results[i, sequences] = 1.
    return results


# 将数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# api方式进行向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 构建网络
models = models.Sequential()
models.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
models.add(layers.Dense(16, activation='relu'))
models.add(layers.Dense(1, activation='sigmoid'))

# 编译模型
# 预设的损失函数、优化器、监控指标
models.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
'''
# 自定义的
models.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy(),
               metrics=[metrics.binary_accuracy()])
'''

# 留出1W个验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# 进行训练
history = models.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

# 绘制训练损失和验证损失
Draw_Diagram.show_loss(history)
Draw_Diagram.show_acc(history)

# 使用测试集进行验证
results = models.evaluate(x_test, y_test)
print("测试集结果:", results)

# 使用网络进行实践
print('预测结果:', models.predict(x_test))
