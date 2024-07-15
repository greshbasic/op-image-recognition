import model
from prediction import predict


if __name__ == "__main__":
    new_model = model.create_model()

    # If you'd like you can import your own photos and have the model
    # take its guess at them :)
    # here is a quick example of how I went about it
    
    # print("which character do you want to predict?")
    # print("1. Luffy")
    # print("2. Sanji")
    # print("3. Law")
    # print("4. Chopper")
    # print("5. Exit")
    # choice = int(input("Enter your choice: "))    
    
    # predict(new_model, choice)
