import pandas as pd
import mysql.connector
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Connect to the database
conn = mysql.connector.connect(
    host="your_host",
    user="your_user",
    password="your_password",
    database="your_database"
)

# Query to get marks data
query = """
SELECT m.student_id, m.subject_id, m.teacher_id, m.mark, s.name as student_name, sub.name as subject_name, t.name as teacher_name
FROM marks m
JOIN students s ON m.student_id = s.id
JOIN subjects sub ON m.subject_id = sub.id
JOIN teachers t ON m.teacher_id = t.id;
"""

# Load data into a DataFrame
df = pd.read_sql(query, conn)
conn.close()

# Feature engineering
df['avg_mark'] = df.groupby('student_id')['mark'].transform('mean')
df['std_mark'] = df.groupby('student_id')['mark'].transform('std')
df['teacher_avg_mark'] = df.groupby(['teacher_id', 'student_id'])['mark'].transform('mean')

# Selecting features
features = df[['avg_mark', 'std_mark', 'teacher_avg_mark']]

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train the Isolation Forest model
model = IsolationForest(contamination=0.1)  # Adjust contamination parameter as needed
model.fit(features_scaled)

# Predict anomalies
df['anomaly'] = model.predict(features_scaled)

# Save results to a CSV file
df.to_csv('favoritism_detection_results.csv', index=False)
