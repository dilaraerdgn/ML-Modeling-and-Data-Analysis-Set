import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# ============================================
# 1) Veri Yükleme
# ============================================
df = pd.read_csv("C:/Users/donusum_islemleri_tamamlanmis_versiyonu.csv")

y = df["charges"]
X = df.drop("charges", axis=1)

# ============================================
# 2) Eğitim - Test Ayrımı
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Ölçekleme (KNN ve SVR için)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 3) Modeller ve Hiperparametre Izgaraları
# ============================================
models = {
    "Linear Regression (Çoklu Doğrusal Regresyon)": {
        "model": LinearRegression(),
        "params": {}
    },
    "KNN Regressor": {
        "model": KNeighborsRegressor(),
        "params": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"]
        }
    },
    "SVR": {
        "model": SVR(),
        "params": {
            "kernel": ["rbf", "linear"],
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"]
        }
    },
    "Random Forest Regressor": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        }
    }
}

# ============================================
# 4) Model Eğitim, Tuning ve Test
# ============================================
results = []

for model_name, mp in models.items():
    print("====================================================")
    print(f"MODEL: {model_name}")
    print("====================================================")

    model = mp["model"]
    params = mp["params"]

    # Skalalanması gereken modeller
    if model_name == "Linear Regression (Çoklu Doğrusal Regresyon)":
        Xtr, Xts = X_train, X_test
    else:
        Xtr, Xts = X_train_scaled, X_test_scaled

    grid = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(Xtr, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    y_pred = best_model.predict(Xts)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("1) En iyi hiperparametreler:", best_params)
    print("2) Test RMSE:", f"{rmse:.4f}")
    print("3) Test R²:", f"{r2:.4f}")
    print()

    results.append([model_name, rmse, r2])

# ============================================
# 5) Modellerin Sonuç Tablosu
# ============================================
df_results = pd.DataFrame(results, columns=["Model", "RMSE", "R²"])
print("\n===============================================")
print(" TÜM MODELLERİN TEST SONUÇ TABLOSU")
print("===============================================")
print(df_results)

# ============================================
# 6) Genel Değerlendirme
# ============================================
print("\n===============================================")
print(" GENEL DEĞERLENDİRME")
print("===============================================")

best_rmse_row = df_results.loc[df_results["RMSE"].idxmin()]

print(f"En başarılı model: {best_rmse_row['Model']}")
print(f"Bu model RMSE açısından en düşük hatayı vermiştir ({best_rmse_row['RMSE']:.4f}).")

print("\nKısa Yorum:")

if "Random Forest" in best_rmse_row["Model"]:
    print("Random Forest, veri içindeki doğrusal olmayan ilişkileri iyi modellediği için daha başarılı olabilir.")
elif "SVR" in best_rmse_row["Model"]:
    print("SVR, kernel yöntemi sayesinde karmaşık ilişkileri yakalayabildiği için daha iyi performans göstermiş olabilir.")
elif "KNN" in best_rmse_row["Model"]:
    print("KNN yerel komşuluk yapısından yararlanarak başarılı tahminler yapmış olabilir.")
else:
    print("Doğrusal regresyon, veri setindeki lineer ilişkinin güçlü olması nedeniyle başarılı olmuş olabilir.")
