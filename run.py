from create_dataset import create_dataset
from train_model import training_model
from data_augmentation import data_augmentation
from test_model import test_model


def choose(i):
    switcher = {
        1: create_dataset,
        2: data_augmentation,
        3: training_model,
        4: test_model,
        5: display_menu,
        6: quit
    }
    func = switcher.get(i, "Invalid Option")()
    return func


def display_menu():
    print("MENU")
    print("***************************************")
    print("1. Create Dataset \n2. Augment the Data \n3. Train the Model \n4. Test the Model \n5. Go Back to the Main Menu \n6. QUIT")
    print("***************************************")



    while True:
        try:
            choice = int(input("Please Enter Your Choice: "))
            if isinstance(choice, int):
                break
        except:
            print("Please use number keys to enter your choice!")

    return choice








if __name__ == '__main__':
    choice = display_menu()
    print(choice)
    choose(choice)



