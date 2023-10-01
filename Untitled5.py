#!/usr/bin/env python
# coding: utf-8

# In[89]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[90]:


# Load your dataset (replace 'data.csv' with your dataset file)
data = pd.read_csv('Train.csv')


# In[91]:


data.columns


# In[78]:


data.head()


# In[92]:


label_encoder = LabelEncoder()
colm_to_encode=['POSTED_BY', 'UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'BHK_OR_RK',
       'SQUARE_FT', 'READY_TO_MOVE', 'RESALE', 'ADDRESS', 'LONGITUDE',
       'LATITUDE', 'TARGET(PRICE_IN_LACS)']
for col in colm_to_encode:
    data[col] = label_encoder.fit_transform(data[col])


# In[93]:


# Define features (X) and target variable (y)
X =data.drop(columns=['TARGET(PRICE_IN_LACS)'])  # Replace with your features
y = data['TARGET(PRICE_IN_LACS)']  # Replace with your target variable


# In[94]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[95]:


# Create a linear regression model
model = LinearRegression()


# In[96]:


# Train the model on the training data
model.fit(X_train, y_train)


# In[84]:


# Make predictions on the test data
y_pred = model.predict(X_test)


# In[97]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[98]:


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()


# In[99]:


coeff_df=pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[ ]:





# In[ ]:




