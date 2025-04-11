
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import shap 
from tqdm import tqdm
from numpy import random
import time


#################################################################################

# Random forest
def fit_random_forest(df):
    # Features e target
    X = df[["sm_s1", "sm_sao", "SIF", "lon", "lat", "Crop_211.0", "Crop_216.0", "Crop_250.0", "Crop_300.0", "Crop_500.0", "Crop_600.0", "Crop_700.0"]]
    y = df["volumetric_soil_water_layer_1"]
    X = X.dropna()
    y = y.loc[X.index]

    # Suddivide il dataset in train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=20)
    model.fit(X_train, y_train)
    
    
    # Predizioni
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calcolo metriche
    rmsd_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r_train, _ = pearsonr(y_train, y_train_pred)
    rmsd_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r_test, p_val = pearsonr(y_test, y_test_pred)

    # Inserisce le previsioni nel dataframe completo
    y_all_pred = model.predict(X)
    df["sm_corrected_rf"] = np.nan
    df.loc[X.index, "sm_corrected_rf"] = y_all_pred

    print("TRAINING SET:")
    print(f"RMSD: {rmsd_train:.4f}, R: {r_train:.4f}")
    print("TEST SET:")
    print(f"RMSD: {rmsd_test:.4f}, R: {r_test:.4f}, p-value: {p_val:.4g}")

    return df, model, rmsd_test, r_test, p_val

# ##################################################################################

# Plot feature importance 

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 4))
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.tight_layout()
    plt.show()
    

#################################################################
# Funzione per l'analisi SHAP su un campione
def shap_analysis(df, rf_model, sample_size=0.1):        
    # df_sample = df.sample(frac=sample_size, random_state=42)
    X_shap = df[["sm_s1", "sm_sao", "SIF", "lon", "lat", 
                        "Crop_211.0", "Crop_216.0", "Crop_250.0", 
                        "Crop_300.0", "Crop_500.0", "Crop_600.0", 
                        "Crop_700.0"]].copy()
    y_shap = df["volumetric_soil_water_layer_1"]
    # Converte tutto in float
    X_shap = X_shap.astype(float)
    rf_model.fit(X_shap, y_shap)

    explainer = shap.Explainer(rf_model, X_shap)
    shap_values = explainer(X_shap)
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.tight_layout()
    plt.savefig(r"D:\Data\saocom\Random_forest\shap_summary_plot.png", dpi=300)  # Salva il file
    plt.close() 
    # plt.show()
    

    # Subplots per dependence plots
    features = ["sm_s1", "sm_sao", "SIF"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    for i, feature in enumerate(features):
        shap.dependence_plot(
            feature, shap_values.values, X_shap,
            ax=axs[i], show=False
        )

    plt.tight_layout()
    plt.savefig(r"D:\Data\saocom\Random_forest\shap_dependence_plots.png", dpi=300)
    # plt.show() 


# #############################################################################

# # Funzione principale

def main(df):
    df_corr, rf_model, rmsd_rf, r_pearson, p_val = fit_random_forest(df)
    
    # Feature importance
    plot_feature_importance(rf_model, ["sm_s1", "sm_sao", "SIF", "lon", "lat", "Crop_211.0", "Crop_216.0", "Crop_250.0", "Crop_300.0", "Crop_500.0", "Crop_600.0", "Crop_700.0"])
    
    # # Esporta le serie corrette
    # outdf = df_corr[["DateTime", "sm_s1", "sm_sao", "SIF", "sm_corrected_rf", "lat", "lon"]]
    # outdf.to_csv(r"D:\Data\saocom\Random_forest\dataset_corrected.py") 
       
    # Esegui l'analisi SHAP
    shap_analysis(df_corr, rf_model)
    
    
    

    

####################################################################################


start_time = time.time()

df = pd.read_csv(r"D:\Data\saocom\Random_forest\dataset.csv")
df_sample = df.sample(frac=0.005, random_state=42)  # Usa il 70% dei dati
df_sample = pd.get_dummies(df_sample, columns=["Crop"], drop_first=True)
print(len(df_sample))
main(df_sample)

print("%s minuti" % ((time.time() - start_time)/60))

# 211 = wheat
# 250 = leguminous XX
# 216 = maize XX
# 300 = woodland
# 500 = grassland
