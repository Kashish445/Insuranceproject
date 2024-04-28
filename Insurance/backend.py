import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import datetime


import opendatasets

opendatasets.download('https://www.kaggle.com/datasets/imtkaggleteam/vehicle-insurance-data/data')
# Load data
insurance_data = pd.read_csv("vehicle-insurance-data/motor_data14-2018.csv")
insurance_data.head()
current_year = datetime.datetime.now().year
# Deduce the age from PROD_YEAR
insurance_data['Age'] = current_year - insurance_data['PROD_YEAR']
insurance_data.head()

# Convert date columns to datetime format
insurance_data['INSR_BEGIN'] = pd.to_datetime(insurance_data['INSR_BEGIN'], format='%d-%b-%y')
insurance_data['INSR_END'] = pd.to_datetime(insurance_data['INSR_END'], format='%d-%b-%y')

# Calculate CUSTOMER_LIFESPAN
insurance_data['CUSTOMER_LIFESPAN'] = (insurance_data['INSR_END'] - insurance_data['INSR_BEGIN']).dt.days
insurance_data["CLAIM_PAID"].fillna(0, inplace=True)
insurance_data['Age'].fillna(insurance_data['Age'].mode()[0], inplace=True)
insurance_data['PREMIUM'].fillna(insurance_data['PREMIUM'].mode()[0], inplace=True)
insurance_data['MAKE'].fillna(insurance_data['MAKE'].mode()[0], inplace=True)

# Calculate revenue_per_customer
insurance_data['revenue_per_customer'] = insurance_data['CLAIM_PAID'] - insurance_data['PREMIUM']
insurance_data.drop(columns=['EFFECTIVE_YR','CCM_TON','PROD_YEAR','INSR_BEGIN','INSR_END','CARRYING_CAPACITY','SEATS_NUM','CCM_TON'], inplace=True)
insurance_data.columns
insurance_data.isnull().sum()
# Group by OBJECT_ID and calculate relevant metrics
sum_data = insurance_data.groupby('OBJECT_ID').agg({'CUSTOMER_LIFESPAN': 'sum', 'revenue_per_customer': 'sum'}).reset_index()

# Calculate average_purchase_frequency
object_id_counts = insurance_data['OBJECT_ID'].value_counts()
average_purchase_frequency = object_id_counts / len(object_id_counts)

# Merge data
data = pd.merge(sum_data, average_purchase_frequency.rename('average_purchase_frequency'), left_on='OBJECT_ID', right_index=True)

# Calculate CLV metrics
data["average_purchase_value"] = data['revenue_per_customer'] / data['OBJECT_ID']
data["customer_value"] = data['average_purchase_value'] / data['average_purchase_frequency']
data["avg_customer_lifetime"] = data['CUSTOMER_LIFESPAN'] / len(object_id_counts)
data["CLV"] = data['customer_value'] * data['avg_customer_lifetime']

# Merge CLV data back to insurance_data
insurance_data_with_clv = pd.merge(insurance_data, data[['OBJECT_ID', 'CLV']], on='OBJECT_ID', how='left')
insurance_data_with_clv.head()
insurance_data_with_clv = insurance_data_with_clv[insurance_data_with_clv['SEX'] != 0]
insurance_data_with_clv.head()
#Transform to M/F
insurance_data_with_clv["SEX"].replace({ 2: 0}, inplace=True)
# Sales_Channel: Make all as Internal
insurance_data_with_clv['Sales_Channel'] = 'internal'
# Claim_Flag: Transform to binary
insurance_data_with_clv['Claim_Flag'] = insurance_data_with_clv['CLAIM_PAID'].apply(lambda x: 0 if x == 0 else 1)
# Define selected features and target variable
selected_features=['SEX','CUSTOMER_LIFESPAN','INSR_TYPE','PREMIUM','USAGE','MAKE','TYPE_VEHICLE','Claim_Flag','Sales_Channel','Age']
X = insurance_data_with_clv[selected_features]
y = insurance_data_with_clv['CLV']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for numeric and categorical features
numeric_features = ['CUSTOMER_LIFESPAN', 'INSR_TYPE', 'PREMIUM', 'Claim_Flag','Age']
numeric_transformer = StandardScaler()

categorical_features = ['SEX', 'USAGE', 'MAKE', 'TYPE_VEHICLE', 'Sales_Channel']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
# Create a list of tuples containing model names and corresponding regressors
regressors = [
    ('Gradient Boosting Regressor', GradientBoostingRegressor()),
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', Ridge()),
    ('Lasso Regression', Lasso())
]

# Iterate through each regressor, fit the pipeline, and print evaluation metrics
for name, regressor in regressors:
    print(f"Model: {name}")
    
    # Create Pipeline
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', regressor)
    ])
    
    # Fit the model pipeline
    model_pipeline.fit(X_train, y_train)
    
    # Make predictions and calculate evaluation metrics
    y_pred = model_pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print evaluation metrics
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    print()

# Create Ridge Regressor pipeline
gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor())
])

# Fit and evaluate the Ridge Regressor model
gb_pipeline.fit(X_train, y_train)
y_pred = gb_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


X = insurance_data_with_clv[selected_features]

# Predict CLV using the trained model
predicted_clv = gb_pipeline.predict(X)

# Add the predicted CLV column to the DataFrame
insurance_data_with_clv['Predicted_CLV_'] = predicted_clv


insurance_data_with_clv.head()

# Save the Gradient Boosting Regressor model to a file
model_filename = 'gb_model1.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(gb_pipeline, file)