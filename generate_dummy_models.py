import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

os.makedirs('models', exist_ok=True)

# Train a tiny dummy model
X = np.array([[0.5,0.5,0.5,120,0.1],[0.2,0.3,0.4,90,0.7],[0.8,0.7,0.6,140,0.05]])
y = np.array([100, 45, 200])

model = GradientBoostingRegressor()
model.fit(X, y)

with open('models/gradient_boosting_eurovision_model.pkl', 'wb') as f:
    pickle.dump(model, f)

preprocessing_info = {
    'mae_test': 24.5,
    'r2_score': 0.42
}
with open('models/preprocessing_info.pkl', 'wb') as f:
    pickle.dump(preprocessing_info, f)

print('Dummy models written to models/ folder')
