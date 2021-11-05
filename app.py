###########################
# BREAST CANCER DETECTION #
###########################

# Based on this video: https://www.youtube.com/watch?v=NSSOyhJBmWY


#### 1. Import libraries ####

# A library is a collection of resources, like pre-written code, used by computer programs
# More info: https://en.wikipedia.org/wiki/Library_(computing)

import numpy                                    # NumPy is a library that offers mathematical functions and adds support for vectors and matrices
import pandas                                   # Pandas is a library for managing and analysing data
import matplotlib.pyplot as plot                # Matplotlib is a library for creating graphical visualizations
import seaborn                                  # Seaborn is a library for data visualization based on Matplotlib (see above)
from sklearn.preprocessing import LabelEncoder  # Scikit-learn is a library for machine learning

# NOTE: All libraries were installed previously by the setup script that I've sent you
# To install a library manually execute 'pip3 install <package-name>' in your terminal


#### 2. Load data ####

# The breast cancer data is stored in the file data.csv (https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
# CSV = 'comma-separated values'
# Watch the video from 4:30 to 8:00 (https://www.youtube.com/watch?v=NSSOyhJBmWY&t=270s)
# NOTE: We don't need to upload the file as we are working on our local computer

data_frame = pandas.read_csv('data.csv')
seaborn.countplot(x='diagnosis', data=data_frame)
plot.show()

# After that go to 13:10 in the video: https://www.youtube.com/watch?v=NSSOyhJBmWY&t=790s



# This is the last line of code which prevents the window from closing until a key is pressed #
input("Press Enter to continue...")

# To run the program execute 'python3 app.py' in your terminal