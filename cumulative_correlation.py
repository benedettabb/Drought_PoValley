
import glob
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr
import numpy as np 
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os


########################################################################################

# Plotta la correlazione cumulata 

def cumulative_corr(df, var, var_meteo, window_size=12):
    correlations = list()
    shifts = list()
    significance = list()

    var = df[var]
    var_meteo = df[var_meteo]
    outdf = pd.DataFrame({"var":var, "var_meteo":var_meteo})
    outdf = outdf.dropna(axis=0)
    outdf["var"] = pd.to_numeric(outdf["var"], errors='coerce')
    outdf["var_meteo"] = pd.to_numeric(outdf["var_meteo"], errors='coerce')

    for window_size in range(2, 13):
        outdf['Cumulative_{}'.format(var_meteo)] = outdf["var_meteo"].rolling(window=window_size, min_periods=1).sum()
        tempdf = pd.DataFrame({"cum":outdf['Cumulative_{}'.format(var_meteo)], "param":outdf["var"]})
        shifts.append(window_size)
        try:
            r, p = pearsonr(tempdf["param"], tempdf["cum"])
        except:
            r, p = np.nan, np.nan

        correlations.append(r)
        significance.append(p)
    final = pd.DataFrame({"Window_months": shifts, "R": correlations, "P": significance})
    final = final[final.P<=0.05]
    try:
        max_value = final.R.iloc[np.argmax(abs(final.R))]
        max_window = final.Window_months.iloc[np.argmax(abs(final.R))]
        return max_value, max_window
    except Exception as e:
        print(e)
        return np.nan, np.nan 

###########################################################################

# Prepara i dataframe 

def to_num (df, coord):
    lon,lat = coord.split("_")
    df["lon"] = pd.to_numeric(df['lon'], errors='coerce')
    df["lat"] = pd.to_numeric(df['lat'], errors='coerce')
    df = df[df.lon == float(lon)]
    df = df[df.lat == float(lat)]
    return df

############################################################################

# Funzione per ottenere i dati unificati e calcolare la correlazione
def main(coord, ndvi_dir, sif_dir, spi_dir, spei_dir, sm_dir):
    try:
        print(coord)
        ndvi_df = to_num(ndvi_dir, coord)
        sif_df = to_num(sif_dir, coord)
        spi_df = to_num(spi_dir, coord)
        spei_df = to_num(spei_dir, coord)
        sm_df = to_num(sm_dir, coord)

        # Unire i dataframe
        merged_df = pd.merge(ndvi_df, sif_df, on='DateTime', how='outer')
        merged_df = pd.merge(merged_df, spi_df, on='DateTime', how='outer')
        merged_df = pd.merge(merged_df, spei_df, on='DateTime', how='outer')
        merged_df = pd.merge(merged_df, sm_df, on='DateTime', how='outer')
        merged_df = merged_df[["DateTime", "NDVI_deseason", "SIF_deseason", "SPI", "SPEI", "sm_deseason"]]
        merged_df = merged_df.rename(columns={"NDVI_deseason":"NDVI", "SIF_deseason":"SIF", "sm_deseason": "SM"})
        
        # Interpola
        for col in merged_df.columns:
            merged_df[col] = merged_df[col].interpolate(method = "linear", limit = 2)

        # Prendi le correlazioni
        max_corr_sif_spi, best_window_sif_spi = cumulative_corr(merged_df, "SIF", "SPI")
        max_corr_sif_spei, best_window_sif_spei = cumulative_corr(merged_df, "SIF", "SPEI")
        max_corr_ndvi_spi, best_window_ndvi_spi = cumulative_corr(merged_df, "NDVI", "SPI")
        max_corr_ndvi_spei, best_window_ndvi_spei = cumulative_corr(merged_df, "NDVI", "SPEI")
        max_corr_sif_sm, best_window_sif_sm = cumulative_corr(merged_df, "SIF", "SM")
        max_corr_ndvi_sm, best_window_ndvi_sm = cumulative_corr(merged_df, "NDVI", "SM")
        max_corr_sm_spi, best_window_sm_spi = cumulative_corr(merged_df, "SM", "SPI")
        max_corr_sm_spei, best_window_sm_spei = cumulative_corr(merged_df, "SM", "SPEI")
        max_corr_ndvi_sif, best_window_ndvi_sif = cumulative_corr(merged_df, "NDVI", "SIF")

        # Aggiungi al dizionario
        data["lat"].append(float(coord.split('_')[1]))
        data["lon"].append(float(coord.split("_")[0]))
        data["max_sif_spi"].append(max_corr_sif_spi)
        data["win_sif_spi"].append(best_window_sif_spi)
        data["max_sif_spei"].append(max_corr_sif_spei)
        data["win_sif_spei"].append(best_window_sif_spei)

        data["max_ndvi_spi"].append(max_corr_ndvi_spi)
        data["win_ndvi_spi"].append(best_window_ndvi_spi)
        data["max_ndvi_spei"].append(max_corr_ndvi_spei)
        data["win_ndvi_spei"].append(best_window_ndvi_spei)
        data["max_ndvi_sif"].append(max_corr_ndvi_sif)
        data["win_ndvi_sif"].append(best_window_ndvi_sif)

        data["max_sif_sm"].append(max_corr_sif_sm)
        data["win_sif_sm"].append(best_window_sif_sm)
        data["max_ndvi_sm"].append(max_corr_ndvi_sm)
        data["win_ndvi_sm"].append(best_window_ndvi_sm)

        data["max_sm_spi"].append(max_corr_sm_spi)
        data["win_sm_spi"].append(best_window_sm_spi)
        data["max_sm_spei"].append(max_corr_sm_spei)
        data["win_sm_spei"].append(best_window_sm_spei)
    except Exception as error:
        print(error)
        for key in data:
            data[key].append(np.nan)
       

# Liste nel dizionario
data = { 
    'lon': [], 'lat': [], 
        "max_sif_spi":[], "win_sif_spi":[],
        "max_sif_spei":[], "win_sif_spei":[],
        "max_ndvi_spi":[], "win_ndvi_spi":[],
        "max_ndvi_spei":[], "win_ndvi_spei":[],
        "max_sif_sm":[], "win_sif_sm":[],
        "max_ndvi_sm":[], "win_ndvi_sm":[],
        "max_sm_spi":[], "win_sm_spi":[],
        "max_sm_spei":[], "win_sm_spei":[],
        "max_ndvi_sif":[], "win_ndvi_sif":[]
}

folder = r"D:\DROUGHT\processing\Montly4MapPlot" 
ndvi_dir = pd.read_csv(os.path.join(folder, "NDVI_monthly_2020_2023.csv"))
sif_dir = pd.read_csv(os.path.join(folder, "SIF_monthly_2020_2023.csv"))
spi_dir = pd.read_csv(os.path.join(folder, "SPI_monthly_2020_2023.csv"))
spei_dir = pd.read_csv(os.path.join(folder, "SPEI_monthly_2020_2023.csv"))
sm_dir = pd.read_csv(os.path.join(folder, "SM_rforest_monthly_2020_2023.csv"))

coords = glob.glob(r"D:\DROUGHT\processing\all\*csv")
[main(Path(c).stem, ndvi_dir, sif_dir, spi_dir, spei_dir, sm_dir)for c in coords]

# Creare il DataFrame finale con i risultati
df_out = pd.DataFrame(data)
df_out.to_csv(r"D:\DROUGHT\results\diff_media\cumulative_correlation_v2.csv")
print(r"Saved in D:\DROUGHT\results\diff_media\cumulative_correlation_v2.csv")








