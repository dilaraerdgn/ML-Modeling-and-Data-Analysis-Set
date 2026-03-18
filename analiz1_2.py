import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Veri seti yükleme
df = pd.read_csv("C:/Users/insurance_1.csv")

# Genel bilgi
print("Veri Seti Bilgisi:")
print(df.info())

# Sayısal değişkenlerin betimsel istatistikleri
print("\nSayısal Değişkenler için Betimsel İstatistikler:")
print(df.describe())

# Kategorik değişkenlerin frekansları
print("\nKategorik Değişken Frekansları:")
for col in df.select_dtypes(include="object").columns:
    print(f"\n{col}:\n", df[col].value_counts())

# Eksik veri analizi
print("\nEksik Veri Analizi:")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
print(pd.DataFrame({"Eksik Sayısı": missing, "Yüzde (%)": missing_percent}))

#GRAFİKLER
#yaş histogram

# 1. Yaş Dağılımı (Histogram)
plt.figure(figsize=(7,5))
plt.hist(df["age"], bins=10, color="skyblue", edgecolor="black")
plt.title("Yaş Dağılımı")
plt.xlabel("Yaş")
plt.ylabel("Kişi Sayısı")
plt.show()

#yaş-smoker histogram

# 1. Histogram (çok Değişkenli Görselleştirme)
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", hue="smoker", bins=20, kde=True, multiple="stack")
plt.title("Yaşa göre sigara kullanımı")
plt.xlabel("Yaş")
plt.ylabel("smoker")
plt.show()

#bmi boxplot

# 2. Boxplot (tek)
plt.figure(figsize=(6, 5))
sns.boxplot(y=df["bmi"], color="lightcoral")
plt.title("Bmi Dağılımı (Vücut kitle indeksi)")
plt.ylabel("BMI")
plt.show()

#bmi-cinsiyet boxplot

# 2. Boxplot (Kategorik vs Sayısal Karşılaştırma)
plt.figure(figsize=(8, 5))
sns.boxplot(x="sex", y="bmi", data=df, palette="pastel")
plt.title("Cinsiyete Göre vücut kitle indeksi (Boxplot)")
plt.xlabel("Cinsiyet")
plt.ylabel("Vücut Kitle İndeksi(bmi)")
plt.show()

#yas-bmi scatter plot

# 3. Scatter Plot (İki Sayısal Değişken Arasındaki İlişki)
plt.figure(figsize=(8, 5))
sns.scatterplot(x="age", y="bmi", data=df, hue="sex", alpha=0.7)
plt.title("cinsiyet ve yaşa bağlı bmi değeri")
plt.xlabel("yas")
plt.ylabel("bmi")
plt.legend(title="Cinsiyet")
plt.show()

#children-charges scatter plot

plt.figure(figsize=(8,5))
sns.scatterplot(x="children", y="charges", data=df, alpha=0.7)
plt.title("Çocuk sayısı ve sigorta ücreti (charges) ilişkisi")
plt.xlabel("Children")
plt.ylabel("Charges")
plt.show()

#bmi-charges ilişkisi

sns.lmplot(x="bmi", y="charges", data=df, aspect=1.5, scatter_kws={"alpha":0.6})
plt.title("BMI ve Charges Arasındaki İlişki", fontsize=14)
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.show()
