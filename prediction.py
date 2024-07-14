from imports import tf, keras, PIL, plt, characters
from PIL import Image

def predict(new_model, choice):
    choice_file = ""
    if choice == 1:
        choice_file = "new_luffy"
    elif choice == 2:
        choice_file = "new_sanji"
    elif choice == 3:
        choice_file = "new_law"
    elif choice == 4:
        choice_file = "new_chopper"
    elif choice == 5:
        return
        
    new_img_path = f"Data/New_Data/{choice_file}.png"
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
    
    return