import model
from imports import tf, keras, PIL, plt, characters
from PIL import Image

new_model = model.create_model()


def prediction():
    new_img_path = "Data/New_Data/new_sanji.png"
    img = Image.open(new_img_path).convert("RGB")
    resized_img = img.resize((180, 180))

    img_arr = keras.preprocessing.image.img_to_array(resized_img)
    img_arr = tf.expand_dims(img_arr, 0)
    img_arr_norm = img_arr / 255.0

    prediction = new_model.predict(img_arr_norm)
    predicted_character_tensor = tf.argmax(prediction, axis=1)
    predicted_character_scalar = predicted_character_tensor.numpy()[0]
    predicted_character = characters[predicted_character_scalar]

    plt.imshow(resized_img)
    plt.axis("off")
    plt.text(0, -20, f"Prediction: {predicted_character}", fontsize=12)
    plt.show()