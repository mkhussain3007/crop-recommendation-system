import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
d = pd.read_csv('Crop_recommendation.csv')

# Preprocess data
crop_dict = {
    "rice": 1, "maize": 2, "jute": 3, "cotton": 4, "coconut": 5, "papaya": 6, "orange": 7,
    "apple": 8, "muskmelon": 9, "watermelon": 10, "grapes": 11, "mango": 12, "banana": 13,
    "pomegranate": 14, "lentil": 15, "blackgram": 16, "mungbean": 17, "mothbeans": 18,
    "pigeonpeas": 19, "kidneybeans": 20, "chickpea": 21, "coffee": 22
}
d["crop_num"] = d["label"].map(crop_dict)
x = d.drop(['label', 'crop_num'], axis=1)
y = d['crop_num']

# Split data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale data
ms = MinMaxScaler()
ms.fit(X_train)
X_train = ms.transform(X_train)
X_test = ms.transform(X_test)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# Train model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Save model
with open('model/crop_model.pkl', 'wb') as f:
    pickle.dump((rfc, ms, sc, crop_dict), f)
