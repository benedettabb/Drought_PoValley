# modello uno 

import pandas as pd
import glob
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import os 
from scipy.stats import pearsonr

# Calcola la distanza tra due coordinate

def distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

#####################################################################################

# Estrae le coordinate dal nome del file

def extract_coordinates(coord_str):
    coord_parts = coord_str.split('_')
    lon = float(coord_parts[0])  
    lat = float(coord_parts[1])  
    return (lat, lon)

#####################################################################################

# Prende il file corrispondente alle coordinate per Sentinel-1 e SAOCOM

def get_identical(directories, coord):
    for dir in directories:
        if Path(dir).stem == coord:
            dataframe = pd.read_csv(dir)
            try:
                dataframe["DateTime"] = pd.to_datetime(dataframe["DateTime"])
            except:
                dataframe["DateTime"] = pd.to_datetime(dataframe["Date-Time"])
    dataframe["DateTime"] = dataframe["DateTime"].dt.normalize()
    return dataframe[dataframe["DateTime"] > '2020-01-01']

#####################################################################

# Interpola il dataframe su base giornaliera

def interpolate(df, date_col, value_col):
    try:
        date_range = pd.date_range(start=df[date_col].min(), end=df[date_col].max(), freq='D')
        interp = interp1d(df.DateTime.astype('int64') // 10**9, df[value_col], kind='linear', fill_value="extrapolate")
        df_daily = pd.DataFrame({'DateTime': date_range})
        df_daily[value_col] = interp(df_daily[date_col].astype('int64') // 10**9)
        return df_daily
    except:
        pass

#############################################################

# Modello di umidità

def soil_moisture_model(x, a, b, c, d):
    sm_s1, sm_sao, sif, month = x
    seasonality = np.cos(2 * np.pi * (month - 1) / 12)
    return (
        a * sm_s1 + 
        b * sm_sao +
        c * sif * seasonality +
        d
    )

#################################################################################

# Ottimizza i parametri sulla base di ERA5

def fit_nonlinear_regression(df):
    X = np.array([df.sm_s1, df.sm_sao, df.SIF, df.month])  # Variabili indipendenti
    y = df.volumetric_soil_water_layer_1     # Target (ERA5)
    # Valori iniziali per i parametri (possiamo affinarli)
    initial_params = [0.5, 0.5, -0.5, 0.5]
    # Stima dei parametri con curve_fit
    popt, pcov = curve_fit(soil_moisture_model, X, y, p0=initial_params)
    # print(f"Parametri ottimizzati: a={popt[0]}, b={popt[1]}, c={popt[2]}, d={popt[3]}")
    # Stima dei valori di umidità del suolo corretta
    df["sm_corrected"] = soil_moisture_model(X,popt[0], popt[1], popt[2], popt[3])
    return df, popt

##########################################################################

# Calcola l'RMSD

def calculate_rmsd(observed, predicted):
    return np.sqrt(np.nanmean((observed - predicted) ** 2))

#########################################################

# Plotta le serie temporali 

def plot_time_series(df, s1_original, sao_original, coord_name, out_fig_dir):
    # Filtrare il df finale (interpolato su ogni giorno) per avere solo le date di acquisizione originali
    df_plt_s1 = df[df['DateTime'].isin(s1_original['DateTime'])] # plottare s1
    df_plt_sao = df[df["DateTime"].isin(sao_original["DateTime"])] #plottare saocom
    df_plt = df[df['DateTime'].isin(s1_original['DateTime']) | df['DateTime'].isin(sao_original['DateTime'])] #plottare l'umidità corretta per le date di s1 e sao

    fig, axs = plt.subplots(4, 1, figsize=(14, 10))  # 4 righe, 1 colonna

    # Primo subplot: ERA5 e SM da S1
    axs[0].plot(df.DateTime, df.volumetric_soil_water_layer_1, label="ERA5", color="blue", alpha=0.5, linewidth = 2)
    axs[0].scatter(df_plt_s1.DateTime, df_plt_s1.sm_s1, label="SM from S1", color="green", s=20, alpha=0.8)
    axs[0].set_ylabel('SM [m3/m3]')
    axs[0].legend()
    axs[0].tick_params(axis='x',  labelbottom=False)

    # Secondo subplot: ERA5 e SM da SAOcom
    axs[1].plot(df.DateTime, df.volumetric_soil_water_layer_1, label="ERA5", color="blue", alpha=0.5, linewidth = 2)
    axs[1].scatter(df_plt_sao.DateTime, df_plt_sao.sm_sao, label="SM from SAOcom", color="red", s=20, alpha=0.8)
    axs[1].set_ylabel('SM [m3/m3]')
    axs[1].legend()
    axs[1].tick_params(axis='x',  labelbottom=False)

    # ERA5 e SIF con due assi verticali
    ax1 = axs[2]
    ax2 = ax1.twinx()  # Aggiungi un secondo asse y
    ax1.plot(df.DateTime, df.volumetric_soil_water_layer_1, label="ERA5", color="blue", alpha=0.5, linewidth = 2)
    ax2.plot(df.DateTime, df.SIF, label="SIF", color="orange", linewidth = 2)
    ax1.set_ylabel('SM [m3/m3]')  # Asse y sinistro per ERA5
    ax2.set_ylabel('SIF')  # Asse y destro per SIF
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.tick_params(axis='x',  labelbottom=False)
    ax2.tick_params(axis='x',  labelbottom=False)

    # ERA5 e SM corretto
    axs[3].plot(df.DateTime, df.volumetric_soil_water_layer_1, label="ERA5", color="blue", alpha=0.5,linewidth = 2)
    axs[3].scatter(df_plt.DateTime, df_plt.sm_corrected, label="SM Corrected", color="purple", s=20, alpha=0.8)
    axs[3].set_ylabel('SM [m3/m3]')
    axs[3].legend()
    axs[3].tick_params(axis='x')

    # Globale
    fig.suptitle('Time Series Analysis for {}'.format(coord_name), fontsize=16)
    fig.tight_layout()
    # plt.show()
    plt.savefig(out_fig_dir, dpi=600)

#############################################################################

# Funzione principale

def main(coord, dirs_s1, dirs_sao, dirs_era5, dirs_sif):
    try:
        print(f"Processing coordinate: {coord}")

        df_s1_original = get_identical(dirs_s1, coord)
        df_sao_original = get_identical(dirs_sao, coord)
        df_sif_original = get_identical(dirs_sif, coord)
        

        # Trova il file ERA5 più vicino
        coord_tuple = extract_coordinates(coord) 
        min_distance = float('inf')
        nearest_file = None
        for era5d in dirs_era5:
            era5_coord = Path(era5d).stem
            era5_coord_tuple = extract_coordinates(era5_coord)
            dist = distance(coord_tuple, era5_coord_tuple)
            if dist < min_distance:
                min_distance = dist
                nearest_file = era5d
        df_era5 = pd.read_csv(nearest_file)
        df_era5["DateTime"] = pd.to_datetime(df_era5["system:time_start"])
        df_era5 = df_era5.loc[df_era5["DateTime"] > "2020-01-01"]

        
        #interpolate do daily
        df_s1 = interpolate(df_s1_original, "DateTime", "vv")
        df_sao = interpolate (df_sao_original, "DateTime", "SM")
        df_sif = interpolate(df_sif_original, "DateTime", "SIF")

        # Uniamo i dati su base giornaliera
        df = df_era5.merge(df_s1, on="DateTime", how="left").merge(df_sif, on="DateTime", how="left").merge(df_sao, on="DateTime", how="left")
        df = df[(df.SIF.notna())&(df.vv.notna())]

        # Aggiungiamo il mese come variabile di stagionalità
        df["month"] = df["DateTime"].dt.month

        df = df.rename(columns={"vv": "sm_s1", "SM": "sm_sao"})
        df = df[df['sm_sao'].notna()]

        # stima dei parametri e correzione dei valori di umidità
        df, params = fit_nonlinear_regression(df)
        a, b, c, d = params
        

        # Scala serie di umidità
        s1_min, s1_max = np.nanmin(df.sm_s1), np.nanmax(df.sm_s1)
        sao_min, sao_max = np.nanmin(df.sm_sao), np.nanmax(df.sm_sao)
        era5min, era5max = np.nanmin(df.volumetric_soil_water_layer_1), np.nanmax(df.volumetric_soil_water_layer_1)
        df["sm_s1"] = era5min + (df.sm_s1 - s1_min) / (s1_max - s1_min) * (era5max - era5min)
        df["sm_sao"] = era5min + (df.sm_sao - sao_min) / (sao_max - sao_min) * (era5max - era5min)
        
        # Calcola la correlazione
        p_s1, _ = pearsonr(df.sm_s1, df.volumetric_soil_water_layer_1)
        p_sao, _ = pearsonr(df.sm_sao, df.volumetric_soil_water_layer_1)
        p_corr, _ = pearsonr(df.sm_corrected, df.volumetric_soil_water_layer_1)

        # Calcola l'errore quadratico
        rmsd_s1 = calculate_rmsd(df.volumetric_soil_water_layer_1, df.sm_s1)
        rmsd_sao = calculate_rmsd(df.volumetric_soil_water_layer_1, df.sm_sao)
        rmsd_corr = calculate_rmsd(df.volumetric_soil_water_layer_1, df.sm_corrected)

        #
        lon_list.append(coord_tuple[1])
        lat_list.append(coord_tuple[0]) 
        rmsd_s1_list.append(rmsd_s1)
        rmsd_sao_list.append(rmsd_sao)
        rmsd_corr_list.append(rmsd_corr)
        p_s1_list.append(p_s1)
        p_sao_list.append(p_sao)
        p_corr_list.append(p_corr)
        a_list.append(a)
        b_list.append(b)
        c_list.append(c)
        d_list.append(d)

        # plot
        # fig_dir = os.path.join(r"D:\DROUGHT\figures\VEGETATION_CORRECTION\time_serie_s1_sao_new_model", "{}.png".format(coord))
        # if not os.path.isfile(fig_dir):
        #     plot_time_series (df, df_s1_original, df_sao_original, coord, fig_dir)

    except:
        print("No Sentinel-1 or SAOCOM observation for this coordinate")
        lon_list.append(np.nan)
        lat_list.append(np.nan) 
        rmsd_s1_list.append(np.nan)
        rmsd_sao_list.append(np.nan)
        rmsd_corr_list.append(np.nan)
        p_s1_list.append(np.nan)
        p_sao_list.append(np.nan)
        p_corr_list.append(np.nan)
        a_list.append(np.nan)
        b_list.append(np.nan)
        c_list.append(np.nan)
        d_list.append(np.nan)

# Lista dei file e delle coordinate
files_s1 = glob.glob(r"C:\Users\Administrator\Documents\DROUGHT\SM_S1\*csv", recursive=True)
files_sao = glob.glob(r"C:\Users\Administrator\Documents\DROUGHT\SM_SAO\*csv", recursive=True)
files_era5 = glob.glob(r"D:\DROUGHT\data\Soil_moisture\Sm_ERA5\ERA5\*csv", recursive=True)
files_sif = glob.glob(r"D:\DROUGHT\data\Vegetation\SIF_csv\8days\*csv", recursive=True)

lon_list = list()
lat_list = list() 
rmsd_s1_list = list()
rmsd_sao_list = list()
rmsd_corr_list = list()
p_s1_list = list()
p_sao_list = list()
p_corr_list = list()
a_list =list()
b_list = list()
c_list = list()
d_list = list()

[main(Path(c).stem, files_s1, files_sao, files_era5, files_sif) for c in files_s1[0:1000]]

params_df = pd.DataFrame({
    "lon": lon_list,
    "lat": lat_list,
    "a": a_list,
    "b": b_list,
    "c": c_list,
    "d": d_list
})
params_df.to_csv(r"D:\DROUGHT\processing\soil_moisture\optimized_params_new_model.csv")


pearson_df = pd.DataFrame({
    "lon": lon_list,
    "lat": lat_list,
    "p_s1": p_s1_list,
    "p_sao": p_sao_list,
    "p_corr": p_corr_list,
    "rmsd_s1": rmsd_s1_list,
    "rmsd_sao": rmsd_sao_list,
    "rmsd_corr": rmsd_corr_list,
})
pearson_df.to_csv(r"D:\DROUGHT\processing\soil_moisture\validation_new_model.csv")