# ============================================
# ÖDEV 5 - BOYUT AZALTMA UYGULAMASI
# En iyi model: Random Forest Regressor
# ============================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


# ============================================
# 1) VERİ SETİNİN YÜKLENMESİ
# ============================================
df = pd.read_csv(
    "C:/Users/donusum_islemleri_tamamlanmis_versiyonu.csv"
)

y = df["charges"]
X = df.drop("charges", axis=1)


# ============================================
# 2) TRAIN - TEST AYRIMI (ÖDEV 4 İLE AYNI)
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================
# 3) ORİJİNAL RANDOM FOREST MODELİ
# ============================================
rf_original = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

rf_original.fit(X_train, y_train)
y_pred_original = rf_original.predict(X_test)

rmse_original = np.sqrt(mean_squared_error(y_test, y_pred_original))
r2_original = r2_score(y_test, y_pred_original)


# ============================================
# 4) FEATURE IMPORTANCE (SADECE TRAIN)
# ============================================
importances = rf_original.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFEATURE IMPORTANCE TABLOSU")
print(feature_importance_df)


# ============================================
# 5) BOYUT AZALTMA (EN ÖNEMLİ %50 ÖZELLİK)
# ============================================
num_features = int(len(feature_importance_df) * 0.5)

selected_features = feature_importance_df["Feature"].iloc[:num_features]

X_train_reduced = X_train[selected_features]
X_test_reduced = X_test[selected_features]

print("\nSEÇİLEN ÖZELLİKLER")
print(selected_features.tolist())


# ============================================
# 6) BOYUT AZALTILMIŞ VERİ İLE MODEL
# ============================================
rf_reduced = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42
)

rf_reduced.fit(X_train_reduced, y_train)
y_pred_reduced = rf_reduced.predict(X_test_reduced)

rmse_reduced = np.sqrt(mean_squared_error(y_test, y_pred_reduced))
r2_reduced = r2_score(y_test, y_pred_reduced)


# ============================================
# 7) SONUÇLARIN KARŞILAŞTIRILMASI
# ============================================
results_df = pd.DataFrame({
    "Veri Seti": ["Orijinal Veri Seti", "Boyutu Azaltılmış Veri Seti"],
    "Test RMSE": [rmse_original, rmse_reduced],
    "Test R²": [r2_original, r2_reduced]
})

print("\n===================================")
print("SONUÇ TABLOSU")
print("===================================")
print(results_df)
