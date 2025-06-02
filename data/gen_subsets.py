from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''
Generate training, validation, and testing subsets.
All the images are resized to 128x128 and converted into grayscale.
The training subset is also augmented.
'''
def gen_subsets(train_folder, val_folder, test_folder):
    train_gen = ImageDataGenerator(rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

    val_test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        train_folder,
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical',
        shuffle=True)

    val_data = val_test_gen.flow_from_directory(
        val_folder,
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

    test_data = val_test_gen.flow_from_directory(
        test_folder,
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical',
        shuffle=False)

    return train_data, val_data, test_data