import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

#veri setini yükleme
df =pd.read_csv("C:/Users/insurance_1.csv")
print("İlk 5 Gözlem:")
print(df.head().T)

# 1. Aykırı Değer Problemi (IQR Yöntemiyle Baskılama)
print("\n--- Aykırı Değer Analizi ---")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
print("\nSayısal Değişkenler:", list(numeric_cols))

def baskila_iqr(df,col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    alt_sinir = Q1 - 1.5 * IQR
    ust_sinir = Q3 + 1.5 * IQR
    # Kaç aykırı değer olduğunu yazdır
    aykiri_sayi = ((df[col] < alt_sinir) | (df[col] > ust_sinir)).sum()
    print(f"{col} değişkeninde {aykiri_sayi} aykırı değer bulundu.")

    # Baskılama işlemi
    df[col] = np.where(df[col] < alt_sinir, alt_sinir,np.where(df[col] > ust_sinir, ust_sinir, df[col]))


for col in numeric_cols:
    baskila_iqr(df,col)



print("\nAykırı değer baskılama işlemi tamamlandı.")
print(df.describe())

# Kaydetme
df.to_csv("C:/Users/aykiri_deger_problemi_cozulmus_versiyonu.csv", index=False)
print("\nAykırı değerleri düzeltilmiş veri kaydedildi: aykiri_deger_problemi_cozulmus_versiyonu.csv")

# 2. Eksik Veri Problemi
df = pd.read_csv("C:/Users/aykiri_deger_problemi_cozulmus_versiyonu.csv")
print("\n--- Eksik Veri Analizi ---")
missing = df.isnull().sum()
print(missing)

missing_percent = (missing / len(df)) * 100
print("\nEksik Veri Oranları (%):")
print(missing_percent)

# Eksik verisi olan sütunları dolduralım
for col in df.columns:
    if df[col].isnull().sum()>0:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"{col} sütunundaki eksikler medyan ile dolduruldu.\n")
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"{col} sütunundaki eksikler mod ile dolduruldu.\n")



print("Doldurma işlemi sonrası eksik veri sayısı:")
print(df.isnull().sum())

df.to_csv("C:/Users/eksik_veri_problemi_cozulmus_versiyonu.csv", index=False)
print("Eksik verileri giderilmiş veri kaydedildi: eksik_veri_problemi_cozulmus_versiyonu.csv")

# 3. Değişken Dönüşümleri (One-Hot Encoding)
df = pd.read_csv("C:/Users/eksik_veri_problemi_cozulmus_versiyonu.csv")
print("\n--- Kategorik Değişken Dönüşümü (One-Hot Encoding) ---")

# Hedef değişken sayısal: 'charges'
target = "charges"
categorical_cols = df.select_dtypes(include="object").columns

print("Kategorik sütunlar:", list(categorical_cols))

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
print("\nDönüşümden Önce Değişken Sayısı:", len(df.columns))
print("Dönüşümden Sonra Değişken Sayısı:", len(df_encoded.columns))

print("\nDönüşüm Sonrası İlk 5 Gözlem:")
print(df_encoded.head().T)

df_encoded.to_csv("C:/Users/donusum_islemleri_tamamlanmis_versiyonu.csv", index=False)
print("\nTüm işlemleri tamamlanmış veri kaydedildi: donusum_islemleri_tamamlanmis_versiyonu.csv")

print("\nVeri ön işleme tamamlandı.")
