import tensorflow as tf 

model_path = 'Training2/NewLayer.h5'

loaded_model = tf.keras.models.load_model(model_path)

def normalized_model(ds):
    # Standarize RGB from 0 to 1
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = ds.map(lambda x, y: (normalization_layer(x), y))
    return normalized_ds


folder_path = 'Database/2Organized'

img_height = 512
img_width = 512

test_ds  = tf.keras.utils.image_dataset_from_directory(
    folder_path,
    validation_split=0.2,
    subset="validation",
    shuffle=True,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=32
)

norm_test = normalized_model(test_ds )

# Evaluate the model on the test dataset
loss, accuracy = loaded_model.evaluate(norm_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy*100:.2f}%')