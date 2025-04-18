import glob 
from pathlib import Path
import numpy as np 
import pandas as pd 
from scipy.interpolate import interp1d
from mpmath import sec
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import concurrent.futures


#####################################################################################

# Media mobile (x è la colonna di un df)

def moving_average(x, w):
    x = x.rolling(window=w, center=False).mean()
    return x.dropna()

############################################################################################

# Rimuove outlaiers

def remove_outlaiers(df):
    lower_percentile = df["S1"].quantile(0.05)
    upper_percentile = df["S1"].quantile(0.95)
    return df[(df["S1"] >= lower_percentile) & (df["S1"] <= upper_percentile)]

###########################################################################################
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

#############################################################################################

# Water Cloud Model 

def water_cloud_model (params, df, plot=False):
    A, B, C = params
    V = df.LAI
    theta_rad = np.deg2rad(38)

    # T^2=exp⁡(-2B∙V∙sec⁡(θ))
    tau = B*V
    tau2 = np.exp(-2*tau*float(sec(theta_rad)) ) 

    # γ_veg^0=A∙V∙cos⁡(θ)(1-T^2)
    veg = A*V*np.cos(theta_rad)*(1-tau2)
    # γ_soil^0  =C ∙exp(SSM)
    soil =  C*np.exp(df.volumetric_soil_water_layer_1)

    #γ^0 =γ_veg^0 +γ_soil^0∙T^2
    df["S1_sim"] = veg+tau2*soil


    if plot:
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5,1,figsize=(10,8))
        ax1.plot(df.DateTime, df.volumetric_soil_water_layer_1, label="ERA5", color="blue")
        ax1.scatter(df.DateTime, df.S1, label="S1", s=5, color="grey")
        ax2.plot(df.DateTime, tau2, label = "Tau2", color="yellow")             
        ax3.plot(df.DateTime, soil, label = "Soil", color="red")
        ax3.plot(df.DateTime, soil*tau2, label = "Vegetation attenuated soil", color="orange")
        ax4.plot(df.DateTime, veg, label = "Vegetation", color="green")
        ax5.scatter(df.DateTime, df["S1_sim"], label="Simulated VV", s=5)
        ax5.scatter(df.DateTime, df.S1, label="S1", s=5, color="grey")

        axes = ax1, ax2, ax3, ax4, ax5
        for a in axes:
            a.legend()
            if not a == ax2: 
                a.set_ylabel("m3/m3")
                a.set_ylim(0,0.5)
            else:
                a.set_ylabel("Unitless")
                a.set_ylim(0,1)

        plt.tight_layout()
        plt.show()
    return df


##################################################################################

def soil_moisture (params, df):
    A, B, C = params
    V = df.LAI
    theta_rad = np.deg2rad(38)
    tau = B*V
    tau2 = np.exp(-2*tau*float(sec(theta_rad)) ) 
    veg = A*V*np.cos(theta_rad)*(1-tau2)
    soil = (df["S1"]- veg)/tau2
    df['SM'] = np.log(soil/C)
    return df



###################################################################################

# Objective 

def objective(params, df):
    ssm_model = water_cloud_model(params, df)
    mae = np.mean(abs((ssm_model["S1_sim"] - ssm_model.S1))) # MAE
    return mae


def objective_soil_moisture(params, df, alpha=1.0, beta=1.0):
    try:
        df = water_cloud_model(params, df)
        df = soil_moisture(params, df)
        df = df.dropna()
        rmse = np.sqrt(np.mean((df['SM'] - df['volumetric_soil_water_layer_1'])**2))
        r, _ = pearsonr(df['SM'], df['volumetric_soil_water_layer_1'])
        loss = alpha * rmse + beta * (1 - r)    
        return loss
    except:
        return np.inf


######################################################################################

# Ottimizzazione parametri 

def optimize_wcm(df):
    initial_guess = [0.5,0.05,0.2]
    result = minimize(objective, initial_guess, args=(df,)) #, bounds=bounds)
    return result.x  # parametri ottimizzati


def optimize_soil_moisture(df):
    initial_guess = [0.5, 0.05, 0.2]
    result = minimize(objective_soil_moisture, initial_guess, args=(df,))
    return result.x


#####################################################################################

# Main

def main (coord, dirs_s1, dirs_sao, dirs_lai, dirs_era5):
    try:
        outname = os.path.join(r"C:\Users\Administrator\Documents\DROUGHT\SM_corrected_WCM_LAI", "{}.csv".format(coord))
        if not os.path.isfile(outname):
            print(f"Processing coordinate: {coord}")

            df_s1 = get_identical(dirs_s1, coord)
            # df_sao = get_identical(dirs_sao, coord)
            df_lai = get_identical(dirs_lai, coord)
            df_lai['LAI'] = pd.to_numeric(df_lai['LAI'], errors='coerce')
            df_lai = df_lai.dropna()
            quality_mask = np.bitwise_and(df_lai['QA'], 0b11)
            df_lai = df_lai[quality_mask == 0].copy()

            df_s1 = df_s1.rename(columns={"vv":"S1"})
            # df_sao = df_sao.rename(columns={"vv":"Sao"})

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

            # Media mobile su tre giorni
            df_s1.S1 = moving_average(df_s1.S1,5)
            # Se ho due osservazioni per una data prendi il valore massimo
            df_s1 = df_s1.groupby(df_s1["DateTime"].copy()).max()
            # Rimuovi gli outlaier
            df_s1 = remove_outlaiers(df_s1)
            df_s1 = df_s1[["S1"]]

            # Scale
            s1_min, s1_max = np.nanmin(df_s1.S1), np.nanmax(df_s1.S1)
            era5min, era5max = np.nanmin(df_era5.volumetric_soil_water_layer_1), np.nanmax(df_era5.volumetric_soil_water_layer_1)
            df_s1.S1 = era5min + (df_s1.S1 - s1_min) / (s1_max - s1_min) * (era5max - era5min)


            # fig, ax1 = plt.subplots(figsize=(12,3))
            # ax1.plot(df_era5.DateTime, df_era5.volumetric_soil_water_layer_1)
            # ax2 = ax1.twinx()
            # ax2.scatter(df_s1.index, df_s1.S1)
            # plt.show()

            # Interpola LAI
            df_lai = interpolate(df_lai, "DateTime", "LAI")

            # Uniamo i dati 
            df = df_era5.merge(df_lai, on="DateTime", how="left") 
            # Per Sentinel-1 la massima tolleranza è di tre giorni
            df = pd.merge_asof(df_s1, df, on="DateTime", direction="nearest", tolerance=pd.Timedelta("3D"))
            df = df.dropna()
            df = df[["DateTime", "volumetric_soil_water_layer_1", "S1", "LAI"]]
            df = df.sort_values("DateTime")

            # Sistema le scale
            df.LAI = era5min + (df.LAI - df.LAI.min()) / (df.LAI.max() - df.LAI.min()) * (era5max - era5min)
            # # Simula il bacskcatter
            df = water_cloud_model([1,0.05,0.2], df, False)
            opt_params = optimize_soil_moisture(df)
            # print("optimized params", opt_params)
            # df = water_cloud_model(opt_params, df, True)
            # r, _ = pearsonr(df.S1_sim, df.S1)
            # rmse = np.sqrt(np.mean((df["S1_sim"] - df.S1)**2))
            # print("Backscatter simulato:", r, rmse)
            # # Inverti per ottenere l'umidità
            df = soil_moisture(opt_params, df)
            df = df.dropna()
            r, p = pearsonr(df.volumetric_soil_water_layer_1, df.SM)
            rmse = np.sqrt(np.mean((df.volumetric_soil_water_layer_1 - df.SM)**2))
            df = df[["DateTime", "SM"]]
            df["SM"] = era5min + (df["SM"] - df["SM"].min()) / (df["SM"].max() - df["SM"].min()) * (era5max - era5min)
            
            df = df.rename(columns={"SM": "sm_corrected_WCM_LAI"})
            
            df.to_csv(outname)
            # print("Umidità:", r, rmse)
            data["lon"].append(coord_tuple[0])
            data["lat"].append(coord_tuple[1])
            data["r"].append(r)
            data["p"].append(p)
            data["rmse"].append(rmse)
            data["a"].append(opt_params[0])
            data["b"].append(opt_params[1])
            data["c"].append(opt_params[2])

    except Exception as e:
        print(e)
        for key in data:
            data[key].append(np.nan)


dirs_s1 = glob.glob(r"C:\Users\Administrator\Documents\DROUGHT\Bs_S1\*csv")
dirs_sao = glob.glob(r"C:\Users\Administrator\Documents\DROUGHT\Bs_SAO\5km_gauss\*csv")
dirs_lai = glob.glob(r"D:\DROUGHT\data\Vegetation\LAI_csv\*csv")
dirs_era5 = glob.glob(r"D:\DROUGHT\data\Soil_moisture\Sm_ERA5\ERA5\*csv")

data = {"lon":[], "lat":[], "r":[], "p":[], "rmse":[], "a":[], "b":[], "c":[]}
[main(Path(coord).stem, dirs_s1, dirs_sao, dirs_lai, dirs_era5) for coord in dirs_s1]
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
        futures = [executor.submit(main, Path(coord).stem, dirs_s1, dirs_sao, dirs_lai, dirs_era5) for coord in dirs_s1]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]


outdf = pd.DataFrame(data)
outdf.to_csv(r"D:\DROUGHT\processing\soil_moisture\water_cloud_model\LAI_results_v2.csv")