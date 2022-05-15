# import the necessary packages
import os

# define the path to the training and testing XML files
TRAIN_PATH = os.path.join("resources",
                          "ibug_300W_large_face_landmark_dataset",
                          "labels_ibug_300W_eyes_train.xml")
TEST_PATH = os.path.join("resources",
                         "ibug_300W_large_face_landmark_dataset",
                         "labels_ibug_300W_eyes_test.xml")

# define the path to the temporary model file
MODEL_PATH = "resources/model/trial_{}.dat"
# define the path to the output CSV file containing the results of
# our experiments
CSV_PATH = "resources/trials.csv"
# define the path to the example image we'll be using to evaluate
# inference speed using the shape predictor
IMAGE_PATH = "resources/marques.jpg"
# define the number of threads/cores we'll be using when trianing our
# shape predictor models
PROCS = -1
# define the maximum number of trials we'll be performing when tuning
# our shape predictor hyperparameters
MAX_TRIALS = 25
