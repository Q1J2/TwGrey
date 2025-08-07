import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

train_generator = datagen.flow_from_directory(
    r"D:\UCM\UCM100\Triplet",
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
     r"D:\UCM\UCM100\Triplet",
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

def create_base_network(input_shape):
    input = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu')(input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    return Model(input, x)

def triplet_loss(y_true, y_pred, alpha=0.4):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss

input_shape = (128, 128, 3)

anchor_input = tf.keras.Input(input_shape)
positive_input = tf.keras.Input(input_shape)
negative_input = tf.keras.Input(input_shape)

base_network = create_base_network(input_shape)

encoded_anchor = base_network(anchor_input)
encoded_positive = base_network(positive_input)
encoded_negative = base_network(negative_input)

output = layers.concatenate([encoded_anchor, encoded_positive, encoded_negative])

triplet_model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=output)

triplet_model.compile(optimizer='adam', loss=triplet_loss)

def extract_features(model, generator):
    features = []
    labels = []
    for batch in generator:
        imgs, lbls = batch
        feats = model.predict(imgs)
        features.append(feats)
        labels.append(lbls)
        if len(features) * generator.batch_size >= generator.samples:
            break
    return np.vstack(features), np.hstack(labels)

X_train, y_train = extract_features(base_network, train_generator)
X_val, y_val = extract_features(base_network, validation_generator)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='macro')
recall = recall_score(y_val, y_pred, average='macro')
f1 = f1_score(y_val, y_pred, average='macro')

print(f"{accuracy:.4f}")
print(f" {precision:.4f}")
print(f" {recall:.4f}")
print(f" {f1:.4f}")
