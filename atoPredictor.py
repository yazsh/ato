import tensorflow as tf
import pandas as pd


def input_function(features,labels):

    # labels = tf.one_hot(labels,depth=3)
    return {"countrycreated": tf.convert_to_tensor(features['countrycreated'].as_matrix(),dtype=tf.float32),
            "countryplaced": tf.convert_to_tensor(features['countryplaced'].as_matrix(), dtype=tf.float32),
            "addresschanged": tf.convert_to_tensor(features['addresschanged'].as_matrix(), dtype=tf.float32),
            "passwordattempts": tf.convert_to_tensor(features['passwordattempts'].as_matrix(), dtype=tf.float32)}, labels

def serving_input_fn():
    inputs = {'countrycreated': tf.placeholder(tf.float32, [None]),
              'countryplaced': tf.placeholder(tf.float32, [None]),
              'addresschanged': tf.placeholder(tf.float32, [None]),
              'passwordattempts': tf.placeholder(tf.float32, [None])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

train = pd.read_csv("/Users/yazen/Desktop/datasets/ATO/ato.csv")

train.columns = ['countrycreated', 'countryplaced', 'addresschanged', 'passwordattempts', 'isFraud']

train_features = train.drop('isFraud', axis=1)

train_labels = train['isFraud']

feature_columns = train_features.columns

my_feature_columns = []
for key in train_features.columns:
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


classifier = tf.estimator.DNNClassifier(
      feature_columns=my_feature_columns, hidden_units=[10, 14, 12, 10], n_classes=2,  )

classifier.train(input_fn=lambda:input_function(train_features, train_labels), steps=200 )

print("evaluating")
classifier.export_savedmodel("/Users/yazen/Desktop/mlprojects/ATO",serving_input_fn)
print(classifier.evaluate(input_fn=lambda:input_function(train_features, train_labels),steps=1))
print("evaluating")
