import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Drop unnecessary columns
df = df.drop(['id'], axis=1)

# Handle missing values
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# Encode categorical features
le = LabelEncoder()
for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    df[col] = le.fit_transform(df[col])

# Save cleaned dataset
df.to_csv("cleaned_stroke_data.csv", index=False)
print("Data cleaned and saved as cleaned_stroke_data.csv")
