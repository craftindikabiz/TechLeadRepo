import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
url = 'https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'
data = pd.read_csv(url)

# Missing part: Display the first few rows of the dataset


# Missing part: Plot the data to visualize the relationship between hours and scores
plt.scatter(data['Hours'])

plt.show()

# Separate features (X) and target (y) from the dataset
X = data[[]]
y = data[]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model using Linear Regression
model = LinearRegression()

# Missing part: Train the model on the training data


# Missing part: Make predictions on the testing data


# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')

# Missing part: Plot the regression line along with the data points
plt.scatter(X, y, color='blue')

plt.show()
