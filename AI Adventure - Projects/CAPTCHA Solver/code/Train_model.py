BATCH_SIZE = 128
EPOCHS = 100


# create dataframes to store historical accuracy and loss for target
hist_train_loss_digit = {i:[] for i in range(NUM_OF_LETTERS)}
hist_test_loss_digit = {i:[] for i in range(NUM_OF_LETTERS)}
hist_train_acc_digit = {i:[] for i in range(NUM_OF_LETTERS)}
hist_test_acc_digit = {i:[] for i in range(NUM_OF_LETTERS)}

hist_train_loss = []
hist_test_loss = []
hist_train_acc = []
hist_test_acc = []

digit_acc = [[] for _ in range(NUM_OF_LETTERS)]
val_digit_acc = [[] for _ in range(NUM_OF_LETTERS)]
loss = []
val_loss = []


history = model.fit(x_train, s_train,
                    batch_size = BATCH_SIZE,
                    epochs = EPOCHS,
                    verbose = 1,
                    validation_data = (x_test, s_test)
                   )
