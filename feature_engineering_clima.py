"""
Athens Climate Feature Engineering and Climate Instability Index (CII)

This script processes historical daily weather observations from Athens
(e.g., temperature, humidity, precipitation) and generates a cleaned,
feature-rich dataset for climate analysis and machine learning.

Main steps:
1. Parse and standardize historical dates with century disambiguation.
2. Clean and repair raw weather data (fix NaNs, unrealistic Tmin/Tmax values).
3. Generate 24 continuous features, including:
   - Basic variables (T, Tmax, Tmin, RH, Precipitation)
   - Temperature-derived metrics (range, midpoint, asymmetry)
   - Temporal lags and rolling statistics
   - Seasonal encodings (sine/cosine transforms)
4. Calculate a novel Climate Instability Index (CII), combining:
   - Thermal instability
   - Moisture instability
   - Weather regime instability
   - Seasonal pattern disruption
5. Normalize the CII to a 0–100 scale.
6. Save the final dataset as CSV and display feature correlations.

Usage:
    Run directly as a script:
        python climate_features.py

Input:
    - "noa_weather_1901_2023.csv" : Raw daily Athens weather data

Output:
    - "athens_climate_features_1901_2023.csv"
      (cleaned dataset with 24 features + CII)

Author: [Your Name]
Date: 2025
"""


import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def parse_athens_dates_working(df):
    """
    Parse ALL dates at once with proper century tracking
    """
    dates = []
    
    for i, date_str in enumerate(df['DATE']):
        try:
            day, month, year = str(date_str).split('-')
            year = int(year)
            
            # Simple rule: assume chronological order
            # Years 00-23 could be 1900-1923 OR 2000-2023
            # Use position in dataset to decide
            
            if year >= 24:  # 24-99 = 1924-1999 (unambiguous)
                full_year = 1900 + year
            elif year == 0:  # 00 = assume 2000 
                full_year = 2000
            else:  # 01-23 = ambiguous
                # Simple heuristic: first half of data = 1900s, second half = 2000s
                if i < len(df) * 0.8:  # First 80% = 1900s
                    full_year = 1900 + year
                else:  # Last 20% = 2000s  
                    full_year = 2000 + year
            
            dates.append(pd.Timestamp(full_year, int(month), int(day)))
            
        except Exception as e:
            print(f"Failed to parse date '{date_str}': {e}")
            dates.append(pd.NaT)
    
    return dates

def create_climate_features(input_file, output_file):
    """
    Create 24 continuous features + Climate Instability Index from basic weather data
    """
    
    print("Loading Athens weather data...")
    df = pd.read_csv(input_file)
    
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("Sample dates:", df['DATE'].head(5).tolist())
    
    # Parse dates with working function
    print("Parsing dates...")
    df['date'] = parse_athens_dates_working(df)
    
    # Check parsing results
    valid_dates = df['date'].notna().sum()
    print(f"Valid dates parsed: {valid_dates} out of {len(df)}")
    
    if valid_dates == 0:
        print("ERROR: No valid dates found! Check your date format.")
        print("Sample DATE values:")
        print(df['DATE'].head(10))
        return None
    
    # Filter and continue
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total records: {len(df)}")
    
    if len(df) == 0:
        print("ERROR: No valid records after date parsing!")
        return None
        
    
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total records: {len(df)}")
    
    # Initialize features dataframe
    features = pd.DataFrame()
    features['DATE'] = df['date'].dt.strftime('%Y-%m-%d')  # Keep as index reference
    
    # 1. Fix cases where Tmax < Tmin
    mask = df['Tmax'] < df['Tmin']
    df.loc[mask, 'Tmax'] = df.loc[mask, 'T'] + (df.loc[mask, 'T'] - df.loc[mask, 'Tmin'])
    
    # 2. Fix NaNs in T when Tmax and Tmin are available and valid
    mask_T_nan = df['T'].isna() & df['Tmax'].notna() & df['Tmin'].notna() & (df['Tmax'] > df['Tmin'])
    df.loc[mask_T_nan, 'T'] = (df.loc[mask_T_nan, 'Tmax'] + df.loc[mask_T_nan, 'Tmin']) / 2

    # 3. Fix NaNs in Tmin (assuming 1 NaN) or if Tmax > T, set Tmin = T - (Tmax - T)
    mask_Tmin_fix = (df['Tmin'].isna()) | ((df['Tmax'] > df['T']) & (df['Tmin'] > df['T']))
    df.loc[mask_Tmin_fix, 'Tmin'] = df.loc[mask_Tmin_fix, 'T'] - (df.loc[mask_Tmin_fix, 'Tmax'] - df.loc[mask_Tmin_fix, 'T'])

    # 4. Set missing precipitation to 0.0
    df['Precipitation'] = df['Precipitation'].fillna(0.0)

    # 5. Fill missing RH with previous cell value
    df['RH'] = df['RH'].fillna(method='ffill')
    
    # Now go to Dataframe and take original+ corrected variables
    T = df['T'].astype(float)
    Tmax = df['Tmax'].astype(float) 
    Tmin = df['Tmin'].astype(float)
    RH = df['RH'].astype(float)
    Precipitation = df['Precipitation'].astype(float)
    
    #Test ranges and existence of NaNs in the dataset
    print("Tmin min/max:", df['Tmin'].min(), df['Tmin'].max())
    print("Tmax min/max:", df['Tmax'].min(), df['Tmax'].max())
    print("T min/max:", df['T'].min(), df['T'].max())
    print("RH min/max:", df['RH'].min(), df['RH'].max())
    print("Precipitation min/max:", df['Precipitation'].min(), df['Precipitation'].max())
    print("Number of NaNs:")
    print(df.isna().sum())
    
    print("Creating features...")
    
    # ===== ORIGINAL VARIABLES (5) =====
    features['T'] = T
    features['Tmax'] = Tmax
    features['Tmin'] = Tmin
    features['RH'] = RH
    features['Precipitation'] = Precipitation
    
    # ===== TEMPERATURE-DERIVED FEATURES (3) =====
    features['temp_range'] = Tmax - Tmin
    features['temp_midpoint'] = (Tmax + Tmin) / 2
    features['temp_asymmetry'] = (T - Tmin) / (Tmax - Tmin + 0.01)  # Avoid division by zero
    features['temp_asymmetry'] = features['temp_asymmetry'].clip(0, 1)  # Keep 0-1 scale
    
    # ===== TEMPORAL LAG FEATURES (6) =====
    features['T_lag1'] = T.shift(1)
    features['T_lag3'] = T.shift(3)
    features['T_lag7'] = T.shift(7)
    features['Tmax_lag1'] = Tmax.shift(1)
    features['RH_lag1'] = RH.shift(1)
    features['precip_lag1'] = Precipitation.shift(1)
    
    # ===== ROLLING STATISTICS (6) =====
    features['T_ma3'] = T.rolling(3, center=True).mean()
    features['T_ma7'] = T.rolling(7, center=True).mean()
    features['T_ma30'] = T.rolling(30, center=True).mean()
    features['RH_ma7'] = RH.rolling(7, center=True).mean()
    features['precip_sum7'] = Precipitation.rolling(7).sum()
    features['precip_sum30'] = Precipitation.rolling(30).sum()
    
    # ===== SEASONAL/TEMPORAL ENCODING (4) - CONTINUOUS =====
    day_of_year = df['date'].dt.dayofyear
    month = df['date'].dt.month
    
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)
    features['day_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)  # Account for leap years
    features['day_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    print("Calculating Climate Instability Index...")
    
    # ===== CLIMATE INSTABILITY INDEX CALCULATION =====
    
    # Component 1: Thermal Instability
    temp_volatility = T.rolling(7).std()
    temp_shock = abs(T - T.shift(1))
    expected_temp_range = (Tmax - Tmin).rolling(30).median()
    temp_range_anomaly = abs((Tmax - Tmin) - expected_temp_range)
    temp_gradient = abs(T.rolling(3).mean() - T.shift(3).rolling(3).mean())
    
    thermal_instability = (
        temp_volatility.fillna(0) * 0.3 +
        temp_shock.fillna(0) * 0.25 + 
        temp_range_anomaly.fillna(0) * 0.25 +
        temp_gradient.fillna(0) * 0.2
    )
    
    # Component 2: Moisture Instability  
    RH_volatility = RH.rolling(7).std()
    RH_shock = abs(RH - RH.shift(1))
    
    # Vapor pressure deficit calculation
    sat_vapor_pressure = 6.11 * np.exp((17.27 * T) / (T + 237.3))
    actual_vapor_pressure = sat_vapor_pressure * (RH / 100)
    vapor_pressure_deficit = sat_vapor_pressure - actual_vapor_pressure
    VPD_instability = vapor_pressure_deficit.rolling(3).std()
    
    precip_volatility = Precipitation.rolling(7).std()
    dry_wet_switches = ((Precipitation > 0.5) != (Precipitation.shift(1) > 0.5)).astype(int)
    
    moisture_instability = (
        RH_volatility.fillna(0) * 0.25 +
        RH_shock.fillna(0) * 0.2 +
        VPD_instability.fillna(0) * 0.3 +
        precip_volatility.fillna(0) * 0.15 +
        dry_wet_switches * 0.1
    )
    
    # Component 3: Weather Regime Instability
    air_mass_indicator = T + (RH - 50) * 0.2
    air_mass_change = abs(air_mass_indicator - air_mass_indicator.shift(1))
    air_mass_persistence = air_mass_indicator.rolling(5).std()
    
    # Temperature-humidity correlation (10-day window)
    temp_RH_correlation = T.rolling(10).corr(RH)#.rolling(10))
    temp_RH_correlation = temp_RH_correlation.replace([np.inf, -np.inf], np.nan).fillna(0)   #to avoid inf or NaN
    correlation_breakdown = abs(temp_RH_correlation - temp_RH_correlation.shift(5))
    correlation_breakdown = correlation_breakdown.replace([np.inf, -np.inf], np.nan).fillna(0) #to avoid inf or NaN
    
    front_proxy = abs(T.diff() + RH.diff() * 0.1)
    
    regime_instability = (
        air_mass_change.fillna(0) * 0.3 +
        air_mass_persistence.fillna(0) * 0.25 +
        correlation_breakdown.fillna(0) * 0.25 +
        front_proxy.fillna(0) * 0.2
    )
    
    # Component 4: Seasonal Pattern Disruption
    # Create seasonal normals using groupby
    seasonal_temp_normal = T.groupby(day_of_year).transform('mean')
    seasonal_RH_normal = RH.groupby(day_of_year).transform('mean')
    
    temp_anomaly = abs(T - seasonal_temp_normal)
    RH_anomaly = abs(RH - seasonal_RH_normal)
    seasonal_transition_factor = np.sin(4 * np.pi * day_of_year / 365.25) ** 2
    
    seasonal_instability = (
        temp_anomaly.fillna(0) * 0.4 +
        RH_anomaly.fillna(0) * 0.3 +
        seasonal_transition_factor * 0.3
    )
    
    # Final Climate Instability Index
    climate_instability_raw = (
        thermal_instability * 0.3 +
        moisture_instability * 0.3 +
        regime_instability * 0.25 +
        seasonal_instability * 0.15
    )
    
    
    # Normalize to 0-100 scale
    min_val = climate_instability_raw.min()
    max_val = climate_instability_raw.max()
    climate_instability_index = ((climate_instability_raw - min_val) / (max_val - min_val)) * 100
    
    features['climate_instability_index'] = climate_instability_index
    
    print("Finalizing dataset...")
    
    # Remove rows with too many NaN values (first 30 days due to rolling windows)
    features_clean = features.dropna()
    
    print(f"Final dataset shape: {features_clean.shape}")
    print(f"Features created: {len(features_clean.columns) - 1}")  # Exclude DATE column
    
    # Save to CSV
    features_clean.to_csv(output_file, index=False)
    print(f"Dataset saved to: {output_file}")
    
    # Display summary statistics
    print("\n=== DATASET SUMMARY ===")
    print(f"Date range: {features_clean['DATE'].min()} to {features_clean['DATE'].max()}")
    print(f"Total records: {len(features_clean)}")
    print(f"Climate Instability Index range: {features_clean['climate_instability_index'].min():.2f} - {features_clean['climate_instability_index'].max():.2f}")
    
    return features_clean

if __name__ == "__main__":
    # Generate the feature dataset
    df = create_climate_features('noa_weather_1901_2023.csv', 'athens_climate_features_1901_2023.csv')
    
    print("\n=== FEATURE CORRELATION SUMMARY ===")
    # Show correlation with target for pruning validation
    target = df['climate_instability_index']
    spearman_corrs = {}

    for col in df.drop(['DATE'], axis=1).columns:
        if col != 'climate_instability_index':
            spearman_corrs[col] = df[col].corr(target, method='spearman')

    correlations = pd.Series(spearman_corrs)

    # Sort by absolute value but preserve signs
    correlations_sorted = correlations.reindex(correlations.abs().sort_values(ascending=False).index)

    print("Top 10 features by correlation strength (with signs):")
    print(correlations_sorted.head(10))
    #print("Top 10 features correlated with Climate Instability Index:")
    #print(correlations.head(10))
    print("\nBottom 5 features (candidates for pruning):")
    print(correlations_sorted.tail(5))
