import glob
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr
import numpy as np 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os 
from scipy import signal


####################################################################

# Prendi il file con le giuste coordinate 

def get_identical(directories, coord, val):
    for dir in directories:
        if Path(dir).stem == coord:
            dataframe = pd.read_csv(dir)
            dataframe['DateTime']=pd.to_datetime(dataframe["DateTime"])
            if val == "SIF":
                dataframe[val] = (dataframe[val]-np.nanmin(dataframe[val]))/(np.nanmax(dataframe[val])-np.nanmin(dataframe[val]))
    return dataframe[(dataframe["DateTime"] > '2020-01-01')&(dataframe["DateTime"] < '2024-01-01')]

##########################################################################

# Plotta le serie temporali 

def timeSeries_plot(merged_df, coord):
    variables = ["SM", "SIF", "NDVI", "SPEI", "SPI"]
    labels = ["SM", "SIF", "NDVI", "SPEI", "SPI"]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    fig, axes = plt.subplots(5, 1, figsize=(10, 6))
    for i, var in enumerate(variables):
        axes[i].scatter(merged_df["DateTime"], merged_df[var], label=labels[i], color=colors[i], s=2)
        axes[i].set_ylabel(labels[i])
        axes[i].legend(loc='upper right')
        if i == 4:  
            axes[i].set_xlabel("DateTime")
        else:
            axes[i].set_xticklabels([])
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(r"D:\DROUGHT\figures\CORRELATION", "{}.png".format(coord)), dpi=300)

#######################################################################################

# Plotta la correlazione tra le variabili

def scatter_plot(merged_df, coord):
    # Lista delle coppie di variabili da plottare
    variable_pairs = [('NDVI', 'SIF'),('SPI', 'SPEI'),('NDVI', 'SPI'),('NDVI', 'SPEI'),
        ('SIF', 'SPI'),('SIF', 'SPEI'), ('SM', 'SPI'),('SM', 'SPEI'),('SM', 'NDVI'),('SM', 'SIF')]
    plt.figure(figsize=(10, 12))
    for i, (var1, var2) in enumerate(variable_pairs, start=1):
        plt.subplot(5, 2, i)
        plt.scatter(merged_df[var1], merged_df[var2])
        plt.xlabel(var1)
        plt.ylabel(var2)
    plt.tight_layout()
    # plt.savefig(os.path.join(r"D:\DROUGHT\figures\CORRELATION", f"{coord}_scatterplot.png"), dpi=300)
    plt.show()


###################################################################################

# Funzione per calcolare la cross-correlazione

def ccf_values(series1, series2):
    p = series1
    q = series2
    p = (p - np.mean(p)) / (np.std(p) * len(p))
    q = (q - np.mean(q)) / np.std(q)  
    valid_indices = ~(np.isnan(p) | np.isnan(q))  
    p = p[valid_indices]
    q = q[valid_indices]
    c = np.correlate(p, q, 'full')
    _, p_value = pearsonr(p, q)
    lags = signal.correlation_lags(len(p), len(q))
    return c, lags * 15, p_value

#############################################################################

# Funzione per il plot della cross-correlazione

def ccf_plot(ax, ccf, lags, title):
    ax.plot(lags, ccf)
    ax.axvline(x=0, color='black', lw=1)
    ax.axhline(y=0, color='black', lw=1)
    ax.axhline(y=np.max(ccf), color='blue', lw=1, linestyle='--', label='highest +/- correlation')
    ax.axhline(y=np.min(ccf), color='blue', lw=1, linestyle='--')
    ax.set(ylim=[-1, 1])
    ax.set_title(title, weight='bold', fontsize=10) 
    ax.set_ylabel('Correlation', fontsize=10)
    ax.set_xlabel('Days', fontsize=10)

#############################################################################

# Funzione per il plot complessivo

def plot_lagged(df, coord):
    variables = [('NDVI', 'SIF'),('NDVI', 'SPI'),('NDVI', 'SPEI'),('SIF', 'SPI'),('SIF', 'SPEI'), ('SPI', 'SPEI'),('SM', 'SPI'),('SM', 'SPEI'),('SM', 'NDVI'),('SM', 'SIF')]
    data["lon"].append(coord.split("_")[0])
    data["lat"].append(coord.split("_")[1])

    # fig, axs = plt.subplots(5, 2, figsize=(10, 12))
    # ax_idx = 0  
    for var1, var2 in variables:
        try:
            ccf, lags, p_value = ccf_values(df[var1], df[var2])
            data[f"max_{var1.lower()}_{var2.lower()}"].append(np.max(ccf))
            data[f"lag_{var1.lower()}_{var2.lower()}"].append(lags[np.argmax(ccf)])
            data[f"p_value_{var1.lower()}_{var2.lower()}"].append(p_value)
            # row, col = divmod(ax_idx, 2)
            # ccf_plot(axs[row, col], ccf, lags, f"{var1} - {var2}")
            # ax_idx += 1
        except:
            data[f"max_{var1.lower()}_{var2.lower()}"].append(np.nan)
            data[f"lag_{var1.lower()}_{var2.lower()}"].append(np.nan)
            data[f"p_value_{var1.lower()}_{var2.lower()}"].append(np.nan)
    

    # plt.tight_layout()
    # plt.show()
    # plt.savefig(os.path.join(r"D:\DROUGHT\figures\CORRELATION", f"{coord}_lagged.png"), dpi=300)

# Funzione per ottenere i dati unificati e calcolare la correlazione
def main(coord, ndvi_dir, sif_dir, spi_dir, spei_dir, sm_dir):
    try:
        # Caricare i dati per ciascuna variabile
        ndvi_df = get_identical(ndvi_dir, coord, "NDVI")
        sif_df = get_identical(sif_dir, coord, "SIF")
        spei_df = get_identical(spei_dir, coord, "SPEI")
        spi_df = get_identical(spi_dir, coord, "SPI")
        sm_df = get_identical(sm_dir, coord, "SM")

        # Unire i dataframe
        merged_df = pd.merge(ndvi_df, sif_df, on='DateTime', how='outer')
        merged_df = pd.merge(merged_df, spei_df, on='DateTime', how='outer')
        merged_df = pd.merge(merged_df, spi_df, on='DateTime', how='outer')
        merged_df = pd.merge(merged_df, sm_df, on='DateTime', how='outer')

        # Chiamare la funzione per il plot delle correlazioni
        # timeSeries_plot(merged_df, coord)
        # scatter_plot(merged_df, coord)
        plot_lagged(merged_df, coord)

    except Exception as error:
        print("An exception occurred:", error)


# Dati aggregati a 15 giorni. Un file csv per ogni coordinata (lon_lat.csv)
ndvi_dir = glob.glob(r"D:\DROUGHT\processing\vegetation\15_days\NDVI\*csv")
sif_dir = glob.glob(r"D:\DROUGHT\processing\vegetation\15_days\SIF\*csv")
spi_dir = glob.glob(r"D:\DROUGHT\processing\meteo\15_days\SPI\*csv")
spei_dir = glob.glob(r"D:\DROUGHT\processing\meteo\15_days\SPEI\*csv")
sm_dir = glob.glob(r"D:\DROUGHT\processing\soil_moisture\15_days\SM\*csv")

# Liste nel dizionario
data = { 
    'lon': [], 'lat': [], 
    'max_ndvi_sif': [], 'max_spi_spei': [], 'max_ndvi_spi': [], 'max_ndvi_spei': [], 'max_sif_spi': [], 'max_sif_spei': [], "max_sm_spi":[], "max_sm_spei":[], "max_sm_ndvi":[], "max_sm_sif":[],
    'lag_ndvi_sif': [], 'lag_spi_spei': [], 'lag_ndvi_spi': [], 'lag_ndvi_spei': [], 'lag_sif_spi': [], 'lag_sif_spei': [], "lag_sm_spi":[], "lag_sm_spei":[], "lag_sm_ndvi":[], "lag_sm_sif":[],
    'p_value_ndvi_sif': [], 'p_value_spi_spei': [], 'p_value_ndvi_spi': [], 'p_value_ndvi_spei': [], 'p_value_sif_spi': [], 'p_value_sif_spei': [], "p_value_sm_spi":[], "p_value_sm_spei":[], "p_value_sm_ndvi":[], "p_value_sm_sif":[]
}

# Elaborazione dei dati per ciascuna coordinata
coords = [Path(f).stem for f in ndvi_dir]
for c in coords[2000:]:
    print("Processing {}".format(c))
    main(c, ndvi_dir, sif_dir, spi_dir, spei_dir, sm_dir)

# Creare il DataFrame finale con i risultati
df_out = pd.DataFrame(data)
df_out.to_csv(r"D:\DROUGHT\results\lagged_correlation_p_f.csv")
