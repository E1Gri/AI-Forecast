import pandas as pd
import numpy as np

def count_mean(df, column_name):
    values = df[df[column_name] != 0.0][column_name].to_numpy()
    
    return values.mean()

def encode_by_mean(df, column_name, edge, mean = None):

    if mean is None:
        mean =  count_mean(df, column_name)

    df1 = df.copy()
    
    for i in range(df1.shape[0]):
        value = float(df1.loc[df1.index[i], column_name])
        if value == 0:
            continue
        elif value - mean > edge:
            df1.loc[df1.index[i], column_name] = 3
        elif value - mean < -edge:
            df1.loc[df1.index[i], column_name] = 1
        else:
            df1.loc[df1.index[i], column_name] = 2

    return (df1, mean)


def encode_data(df, means = None):
    df1 = df.copy()

    if means is None:
        df1, mean_rain = encode_by_mean(df1, "rain_sum (mm)", 1.5)
        df1, mean_pressure = encode_by_mean(df1, "surface_pressure_mean (hPa)", 4.2)
    else:
        df1, mean_rain = encode_by_mean(df1, "rain_sum (mm)", 1.5, means[0])
        df1, mean_pressure = encode_by_mean(df1, "surface_pressure_mean (hPa)", 4.2, means[1])

    column_name = "wind_speed_10m_mean (km/h)"
    for i in range(df1.shape[0]):
        value = float(df1.loc[df1.index[i], column_name])
        
        if value <= 1.6:
            df1.loc[df1.index[i], column_name] = 0
        elif value <= 19.3:
            df1.loc[df1.index[i], column_name] = 1
        elif value <= 38.6:
            df1.loc[df1.index[i], column_name] = 2
        else:
            df1.loc[df1.index[i], column_name] = 3
    
    column_name = "temperature_2m_mean (°C)"
    for i in range(df1.shape[0]):
        value = float(df1.loc[df1.index[i], column_name])
        
        if value <= -20:
            df1.loc[df1.index[i], column_name] = 0
        elif value <= -5:
            df1.loc[df1.index[i], column_name] = 1
        elif value < 5:
            df1.loc[df1.index[i], column_name] = 2
        elif value <= 10:
            df1.loc[df1.index[i], column_name] = 3
        elif value <= 20:
            df1.loc[df1.index[i], column_name] = 4
        else:
            df1.loc[df1.index[i], column_name] = 5

    column_name = "wind_direction_10m_dominant (°)"
    for i in range(df1.shape[0]):
        value = float(df1.loc[df1.index[i], column_name]) % 360
    
        if value < 30 or value >= 330:
            df1.loc[df1.index[i], column_name] = 0
        elif value < 60:
            df1.loc[df1.index[i], column_name] = 1
        elif value < 120:
            df1.loc[df1.index[i], column_name] = 2
        elif value < 150:
            df1.loc[df1.index[i], column_name] = 3
        elif value < 210:
            df1.loc[df1.index[i], column_name] = 4
        elif value < 240:
            df1.loc[df1.index[i], column_name] = 5
        elif value < 300:
            df1.loc[df1.index[i], column_name] = 6
        else:
            df1.loc[df1.index[i], column_name] = 7
    
    return [df1, [mean_rain, mean_pressure]]

def add_lags(df, max_lags, drop_time = True):
    df2 = df.copy()

    lags = {}

    for i in range(1, max_lags + 1):
        lags[f'rain_shift{i}'] = (
            df2['rain_sum (mm)'].shift(i)
        )

        lags[f'wind_dir_shift{i}'] = (
            df2['wind_direction_10m_dominant (°)'].shift(i)
        )

        lags[f'wind_speed_shift{i}'] = (
            df2['wind_speed_10m_mean (km/h)'].shift(i)
        )

        lags[f'pressure_shift{i}'] = (
            df2['surface_pressure_mean (hPa)'].shift(i)
        )

        lags[f'temperature_shift{i}'] = (
            df2['temperature_2m_mean (°C)'].shift(i)
        )

    lag_df = pd.DataFrame(lags, index=df2.index)

    df2 = pd.concat([df2, lag_df], axis=1)

    if drop_time:
        df2 = df2.drop(columns=["time"])
    else:
        df2["time"] = pd.to_datetime(df2["time"])
        df2["time"] = df2["time"].dt.day_of_year
    
    df2 = df2.drop(columns = ['wind_direction_10m_dominant (°)', 'wind_speed_10m_mean (km/h)', 'wind_direction_10m_dominant (°)', 'surface_pressure_mean (hPa)', 'temperature_2m_mean (°C)'])
    df2 = df2.dropna()
    df2 = df2.rename(columns={'rain_sum (mm)': 'target'})

    return df2