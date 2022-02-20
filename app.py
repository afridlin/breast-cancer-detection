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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

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
# print(data_frame['diagnosis'].value_counts())

# Visualize the numbers
# seaborn.countplot(data_frame['diagnosis'], label='count')
# plot.show()

# Prepare the data for machine learning
encoder = LabelEncoder()
data_frame.iloc[:,1] = encoder.fit_transform(data_frame.iloc[:,1].values) # all rows, column 1 (diagnosis); starts from 0; 0 would be column 'id'

# Correlate the first 4 columns (factors) to each other
# seaborn.pairplot(data_frame.iloc[:,1:5], hue='diagnosis')
# plot.show()

# Get the correlation of the columns
# print(data_frame.iloc[:,1:12].corr())

# Continue at https://youtu.be/NSSOyhJBmWY?t=1756

plot.figure(figsize=(10,10))
plot.gca().set_aspect('equal')
seaborn.heatmap(data_frame.iloc[:,1:12].corr(), annot=True, fmt='.0%')
plot.subplots_adjust(bottom=0.25)
plot.show()

# Continue at https://youtu.be/NSSOyhJBmWY?t=2040

x = data_frame.iloc[:, 2:31].values
y = data_frame.iloc[:, 1].values

# split the data into 75 % training data and 25 % testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# scale the data (Feature Scaling)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

def models(x_train, y_train):
    log = LogisticRegression(random_state = 0)
    log.fit(x_train, y_train)

    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(x_train, y_train)

    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(x_train, y_train)

    # print('Logistic Regression Training Accuracy', log.score(x_train, y_test))
    # print('Decision Tree Classifier Training Accuracy', tree.score(x_train, y_test))
    # print('Random Forest Classifier Training Accuracy', forest.score(x_train, y_test))

    return log, tree, forest

model = models(x_train, y_train)

cm = confusion_matrix(y_test, model[0].predict(x_test))

tn = cm[0][0]
tp = cm[1][1]
fn = cm[1][0]
fp = cm[0][1]

print(cm)
print('Testing Accuracy: ', (tp + tn) / (tp + tn + fn + fp))

# Continue at https://youtu.be/NSSOyhJBmWY?t=3203