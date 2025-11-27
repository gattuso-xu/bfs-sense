import scipy.io as scio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


def plot_confusion_matrix(y_true, y_pred, classes=['wave', 'push', 'circle']):
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percentage = np.nan_to_num(cm_percentage)

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=False)

    for i in range(len(classes)):
        for j in range(len(classes)):
            count = cm[i, j]
            percentage = cm_percentage[i, j] * 100
            ax.text(j + 0.5, i + 0.5, f'{count}\n({percentage:.2f}%)', ha='center', va='center', fontsize=12,
                    color='black')

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


def history_plot(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()


data_1 = abs(scio.loadmat('D:\\BFI activity\\timeSegment\\wave\\wave.mat')['mergedData'])
data_3 = abs(scio.loadmat('D:\\BFI activity\\timeSegment\\push\\push.mat')['mergedData'])
data_4 = abs(scio.loadmat('D:\\BFI activity\\timeSegment\\circle\\circle.mat')['mergedData'])

print(f"Class 1 shape: {data_1.shape}")
print(f"Class 3 shape: {data_3.shape}")
print(f"Class 4 shape: {data_4.shape}")

# 创建标签
labels_1 = np.zeros((data_1.shape[0], 1))  # 类别 0
labels_3 = np.full((data_3.shape[0], 1), 1)  # 类别 1
labels_4 = np.full((data_4.shape[0], 1), 2)  # 类别 2


data = np.concatenate((data_1, data_3, data_4), axis=0)
labels = np.concatenate((labels_1, labels_3, labels_4), axis=0)
labels = to_categorical(labels, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], 112, 4, 234)
X_test = X_test.reshape(X_test.shape[0], 112, 4, 234)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(112, 4, 234)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=32, shuffle=True, verbose=1)
model.save("D:\pythonProject\AdversarialAttack\\activity_model_three_class.h5", options='utf-8')

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")

history_plot(history)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

plot_confusion_matrix(y_true, y_pred_classes, ['wave', 'push', 'circle'])
