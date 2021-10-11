def plot_diagram(digit_acc_now, val_digit_acc_now, loss_now, val_loss_now):
    global digit_acc, val_digit_acc, loss, val_loss
    
    for i in range(NUM_OF_LETTERS):
        digit_acc[i].extend(digit_acc_now[i])
        val_digit_acc[i].extend(val_digit_acc_now[i])
    loss.extend(loss_now)
    val_loss.extend(val_loss_now)
    
    # generate accuracy of recognizing and predicting each individual character in validation.
    # may add more columns depending on the number of letters.
    # The number of letters used per captcha is relatively small, 
    # so a dynamically changing array is not necessary
    for i in range(NUM_OF_LETTERS):
        s = {0:'First', 
             1:'Second',
             2:'Third',
             3:'Fourth'
             # 4:'Fifth',
             # 5: 'Sixth'
             }[i]
        plt.plot(digit_acc[i], label = '%s Digit Test' % s)

    # Plot validation accuracy over time
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    #generate accuracy of recognizing and predicting each individual character in training set.
    for i in range(NUM_OF_LETTERS):
        s = {0:'First', 
             1:'Second', 
             2:'Third', 
             3:'Fourth' 
             #4:'Fifth', 
             #5: 'Sixth'
             }[i]
        plt.plot(val_digit_acc[i], label = '%s Digit Train' % s)

    # Plot Accuracy over time on Train Set
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()



    # Plot training & validation loss values
    plt.plot(val_loss, label = 'Train')
    plt.plot(loss, label = 'Test')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()



plot_diagram(    
    [history.history['digit%d_accuracy' % i] for i in range(NUM_OF_LETTERS)],
    [history.history['val_digit%d_accuracy' % i] for i in range(NUM_OF_LETTERS)],
    history.history['loss'],
    history.history['val_loss'],
)


# Save model and weights
save_dir = os.path.join('/content/', 'saved_models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_name = 'keras_trained_model.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at' + model_path)
