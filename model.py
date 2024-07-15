# saving the trouble of waiting for hefty imports if the data folder does not exist anyway
# -------------------------------------------------------
from pathlib import Path
data = Path("Data/Training_Data")
if not data.exists():
    exit("Data folder does not exist")
    
from imports import tf, plt, keras, layers, Sequential, randint, characters, ReduceLROnPlateau, regularizers
# -------------------------------------------------------
def create_model():
    batch = 32
    img_h = 180
    img_w = 180

    random_seed = randint(1, 1001)
    training_dataset = tf.keras.utils.image_dataset_from_directory(
        data,
        validation_split=0.15,
        subset="training",
        seed=random_seed,
        image_size=(img_h, img_w),
        batch_size=batch
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        data,
        validation_split=0.15,
        subset="validation",
        seed=random_seed,
        image_size=(img_h, img_w),
        batch_size=batch
    )

    plt.figure(figsize=(10, 10))
    for images, labels in training_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(characters[labels[i]])
            plt.axis("off")

    AUTOTUNE = tf.data.AUTOTUNE

    training_dataset = training_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.Rescaling(1./255)
    normalized_ds = training_dataset.map(lambda x, y: (normalization_layer(x), y))
    _ , keras.labels_batch = next(iter(normalized_ds))

    num_classes = len(characters)
    
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(img_h, img_w, 3)),
        layers.Conv2D(32, 3, padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        
        layers.Conv2D(64, 3, padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        
        layers.Conv2D(64, 3, padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, 3, padding='same'),
        layers.Activation('relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        
        layers.Flatten(),
        layers.Dense(256),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
    model.summary()
    
    epochs=15
    reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.005)
    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[reduce_learning_rate]
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    
    return model