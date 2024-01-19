import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
file_path = r'combined_data_file_path'
data = pd.read_csv(file_path)

# Selecting the desired features as X and 'Std_Avg_Beta' as y
mfcc_columns = ['Max_0','Min_0','Mean_0','Std_0','Max_1','Min_1','Mean_1','Std_1',
                'Max_2','Min_2','Mean_2','Std_2',
                'Max_3','Min_3','Mean_3','Std_3',
                'Max_4','Min_4','Mean_4','Std_4',
                'Max_5','Min_5','Mean_5','Std_5',
                'Max_6','Min_6','Mean_6','Std_6',
                'Max_7','Min_7','Mean_7','Std_7',
                'Max_8','Min_8','Mean_8','Std_8',
                'Max_9','Min_9','Mean_9','Std_9',
                'Max_10','Min_10','Mean_10','Std_10',
                'Max_11','Min_11','Mean_11','Std_11',
                'Max_12','Min_12','Mean_12','Std_12']
heart_rate_columns = ['Max_value', 'Min_value', 'Mean_value', 'Std_value']
selected_columns = heart_rate_columns + mfcc_columns
data['Diff_Avg_Beta'] = data['Max_Avg_Beta'] - data['Min_Avg_Beta']
yval_column = ['Diff_Avg_Beta']

# Calculate the correlation matrix including the target variable
correlation_matrix = data[selected_columns + yval_column].corr()  #Std_Avg_Beta, Max_Avg_Beta

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Calculate correlation of selected features with 'Std_Avg_Beta'
feature_correlation = data[selected_columns].corrwith(yval_column)  #Std_Avg_Beta, Max_Avg_Beta

# Print out the correlation values
print("Correlation to Max_Avg_Beta:")     ##Std_Avg_Beta, Max_Avg_Beta
print(feature_correlation)

# Assuming 'Std_Avg_Beta' is the target variable we want to predict
X = data[selected_columns]
y = yval_column                ##Std_Avg_Beta, Max_Avg_Beta

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor model
rf = RandomForestRegressor(random_state=42)

# Train the model on the training data
rf.fit(X_train, y_train)

# Predict on the test data
y_pred = rf.predict(X_test)

# Calculate the evaluation metrics
rmse = mean_squared_error(y_test, y_pred, squared=False)  # Set squared=False for RMSE
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print(f'RMSE: {rmse}')
print(f'R2 Score: {r2}')
