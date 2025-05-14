import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

dem = rasterio.open('data/dem.tif')
ortho = rasterio.open('data/ortho.tif')
dtm = rasterio.open('data/dtm.tif')

dem_arr = dem.read(1)
ortho_arr = ortho.read([1, 2, 3, 4, 5, 6])  
dtm_arr = dtm.read(1)

plots_1 = gpd.read_file('data/plots_1.shp')
plots_2 = gpd.read_file('data/plots_2.shp')

elevation = np.where(dem_arr <= 0, np.nan, dem_arr)
masked_thermal = np.where(ortho_arr[5] <= 0, np.nan, ortho_arr[5] / 100 - 273.15)

with np.errstate(divide='ignore', invalid='ignore'):
    ndvi = (ortho_arr[3] - ortho_arr[0]) / (ortho_arr[3] + ortho_arr[0])
    ndvi = np.where(np.isnan(ndvi), 0, ndvi)

def compute_zonal_stats(plots, data, affine):
    import rasterstats as rs
    plot_zs = rs.zonal_stats(plots, data, affine=affine, stats="mean", nodata=np.nan, geojson_out=True)
    return gpd.GeoDataFrame.from_features(plot_zs)

plots_1['NDVI_mean'] = compute_zonal_stats(plots_1, ndvi, dem.transform)['mean']
plots_1['thermal_mean'] = compute_zonal_stats(plots_1, masked_thermal, dem.transform)['mean']
plots_1['elevation_mean'] = compute_zonal_stats(plots_1, elevation, dem.transform)['mean']
plots_1['dtm_mean'] = compute_zonal_stats(plots_1, dtm_arr, dem.transform)['mean']   

plots_1 = plots_1.dropna()
features = plots_1[['NDVI_mean', 'thermal_mean', 'elevation_mean', 'dtm_mean']]

healthy_mask = (plots_1['NDVI_mean'] >= 0.4) & (plots_1['NDVI_mean'] <= 0.8)
plots_1['synthetic_target'] = np.where(healthy_mask, 1, 0)

class_counts = plots_1['synthetic_target'].value_counts()
print(f"Class distribution:\n{class_counts}")

from sklearn.utils import resample

minority_class = plots_1[plots_1['synthetic_target'] == 1]
majority_class = plots_1[plots_1['synthetic_target'] == 0]
majority_class_downsampled = resample(majority_class, 
                                      replace=False, 
                                      n_samples=len(minority_class), 
                                      random_state=42)
balanced_data = pd.concat([minority_class, majority_class_downsampled])
features = balanced_data[['NDVI_mean', 'thermal_mean', 'elevation_mean', 'dtm_mean']]
target = balanced_data['synthetic_target']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Sigmoid activation function for binary classification
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=32)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy on test data: {accuracy}")
predictions = (model.predict(X_test) > 0.5).astype("int32")
accuracy_score_val = accuracy_score(y_test, predictions)
precision_score_val = precision_score(y_test, predictions)
recall_score_val = recall_score(y_test, predictions)
f1_score_val = f1_score(y_test, predictions)
roc_auc_score_val = roc_auc_score(y_test, model.predict(X_test))

print(f"Accuracy Score: {accuracy_score_val}")
print(f"Precision Score: {precision_score_val}")
print(f"Recall Score: {recall_score_val}")
print(f"F1 Score: {f1_score_val}")
print(f"ROC AUC Score: {roc_auc_score_val}")
conf_matrix = confusion_matrix(y_test, predictions)

# Plot the confusion matrix using Seaborn's heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Negative', 'Predicted Positive'], 
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()