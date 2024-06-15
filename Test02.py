import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv(url, header=None, names=column_names)

# Missing part: Display the first few rows of the dataset


# Missing part: Separate features (X) and target (y) from the dataset
X = data
y = data

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (fit on training data, transform both training and testing data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the k-NN model
k = 3  # Number of neighbors
model = KNeighborsClassifier(n_neighbors=k)

# Missing part: Train the model on the training data
model.fit(x, y)

# Missing part: Make predictions on the testing data
y_pred = model

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)

# Missing part: Print the classification report to provide more details about model performance
from sklearn.metrics import classification_report
print(classification_report(y, y))
