###########################
# BREAST CANCER DETECTION #
###########################

# Based on this video: https://www.youtube.com/watch?v=NSSOyhJBmWY


#### 1. Import libraries ####

# A library is a collection of resources, like pre-written code, used by computer programs
# More info: https://en.wikipedia.org/wiki/Library_(computing)

import numpy                                         # NumPy is a library that offers mathematical functions and adds support for vectors and matrices
import pandas                                        # Pandas is a library for managing and analysing data
import matplotlib.pyplot as plot                     # Matplotlib is a library for creating graphical visualizations
import seaborn                                       # Seaborn is a library for data visualization based on Matplotlib (see above)

# Scikit-learn is a library for machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# NOTE: All libraries were installed previously by the setup script that I've sent you
# To install a library manually execute 'pip3 install <package-name>' in your terminal


#### 2. Load data ####

# The breast cancer data is stored in the file data.csv (https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
# CSV = 'comma-separated values'
# Watch the video from 4:30 to 8:00 (https://www.youtube.com/watch?v=NSSOyhJBmWY&t=270s)
# NOTE: We don't need to upload the file as we are working on our local computer

data_frame = pandas.read_csv('data.csv')

# After that go to 13:10 in the video: https://www.youtube.com/watch?v=NSSOyhJBmWY&t=790s

# Get the number of malignant (M) and benign (B) cells
print(data_frame['diagnosis'].value_counts())

# Visualize the numbers
seaborn.countplot(data_frame['diagnosis'], label='count')
plot.show()

# Prepare the data for machine learning
encoder = LabelEncoder()
data_frame.iloc[:,1] = encoder.fit_transform(data_frame.iloc[:,1].values) # all rows, column 1 (diagnosis); starts from 0; 0 would be column 'id'

# Correlate the first 4 columns (factors) to each other
seaborn.pairplot(data_frame.iloc[:,1:5], hue='diagnosis')
plot.show()

# Get the correlation of the columns
print(data_frame.iloc[:,1:12].corr())

# Continue at https://youtu.be/NSSOyhJBmWY?t=1756