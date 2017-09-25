# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import os
import time

def generate_timestamp():
    timestring = time.strftime("%Y_%m_%d-%H_%M_%S")
    print ("time stamp generated: "+timestring)
    return timestring

timestr = generate_timestamp()

seed = 7                                                                        # Fix random seed for reproducibility
numpy.random.seed(seed)

dataset = numpy.loadtxt("/home/info/pima-indians-diabetes.data", delimiter=",") # load dataset

X = dataset[:,0:8]                                                              # Split into input (X) and output (Y) variables
Y = dataset[:,8]

def save_model(model, file_loc):
    file_pointer = os.path.join(file_loc[0]+"//trained_"+ timestr)
    model.save_weights(os.path.join(file_pointer + ".model"))
    # serialize model to JSON
    model_json = model.to_json()
    with open(file_pointer+"_model.json", "w") as json_file:
        json_file.write(model_json)
    print ("Saved the trained model weight to: " + 
           str(os.path.join(file_pointer + ".model")))
        

model = Sequential()                                                            # Create model
model.add(Dense(12, input_dim=8, 
                kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy',                                       # Compile model
              optimizer='adam', metrics=['accuracy'])                           

history = model.fit(X, Y, validation_split=0.33, 
                    epochs=150, batch_size=10, verbose=1)                       # Fit the model

save_model(model,  ['/home/info'])

# list all data in history
print(history.history.keys())

def plot_training(history):
  output_loc = '/home/info/'
  output_file_acc = os.path.join(output_loc+"//training_plot_acc_"+
                                 timestr + ".png")
  output_file_loss = os.path.join(output_loc+"//training_plot_loss_"+
                                  timestr +".png")
  
  fig_acc = plt.figure()
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  fig_acc.savefig(output_file_acc, dpi=fig_acc.dpi)
  print ("Successfully created model training accuracy plot: " + 
         str(output_file_acc))
  plt.close()

  fig_loss = plt.figure()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  fig_loss.savefig(output_file_loss, dpi=fig_loss.dpi)
  print ("Successfully created model training loss function plot: " + 
         str(output_file_loss))
  plt.close()  
  
plot_training(history)