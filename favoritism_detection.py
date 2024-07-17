import logging
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to the database
DATABASE_USERNAME = 'root'
DATABASE_PASSWORD = ''
DATABASE_HOST = 'localhost'
DATABASE_NAME = 'student_management'

engine = create_engine(f'mysql+pymysql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_NAME}')

# Load data
marks = pd.read_sql('SELECT * FROM marks', engine)

# Calculate average marks for each student
student_avg_marks = marks.groupby('student_id')['marks'].mean().reset_index()

# Merge average marks with original data
marks = marks.merge(student_avg_marks, on='student_id', suffixes=('_original', '_avg'))

# Calculate deviation from average for each mark
marks['deviation'] = marks['marks_original'] - marks['marks_avg']

# Log average marks calculation
logging.debug("Average Marks Calculation:")
logging.debug(student_avg_marks.head())

# Log merged data
logging.debug("\nMerged Data with Average Marks and Deviation:")
logging.debug(marks.head())

# Use Isolation Forest for anomaly detection
def detect_anomalies(df):
    X = df[['deviation']]

    # Convert average marks to integer for random_state
    df['random_state'] = df['marks_avg'].apply(lambda x: hash(x) % (2**32))

    # Log isolation forest input
    logging.debug("\nIsolation Forest Input (X):")
    logging.debug(X.head())

    logging.debug("\nRandom State Values:")
    logging.debug(df['random_state'].head())

    # Initialize Isolation Forest model
    isolation_forest = IsolationForest(contamination=0.1, random_state=df['random_state'].iloc[0])

    # Fit model and predict anomalies
    anomalies = isolation_forest.fit_predict(X)

    # Log anomalies detected
    logging.debug("\nAnomalies Detected:")
    logging.debug(anomalies)

    # Get indices of anomalies
    anomaly_indices = np.where(anomalies == -1)[0]

    # Get corresponding student_ids, subject_ids, marks_original, id, and teacher_id
    anomalous_students = df.iloc[anomaly_indices][['id', 'student_id', 'subject_id', 'marks_original', 'teacher_id']]

    return anomalous_students

anomalous_students = detect_anomalies(marks)

# Log number of anomalies found
logging.debug(f"\nNumber of anomalies found: {len(anomalous_students)}")

# Log anomalous students details
logging.debug("Anomalous Students Details:")
logging.debug(anomalous_students.head())

# Output anomalous students as JSON
anomalous_students_json = anomalous_students.to_json(orient='records')
logging.debug("Anomalous Students JSON Output:")
logging.debug(anomalous_students_json)

# Print JSON output for verification
print(anomalous_students_json)  
