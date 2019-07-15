from keras.datasets import imdb

# 加载IMDB数据集  num_word表示保留词频为前10000的单词
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

