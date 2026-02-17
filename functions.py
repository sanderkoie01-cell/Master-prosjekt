import numpy as np
import pandas as pd
import turbine_models
from turbine_models.parser import Turbines
from scipy.stats import norm

def map_turbine_model(start_year: int, installation_type: str):
    """
    Map a wind farm's start year to an appropriate turbine model AND 
    a representative hub height.
    
    Returns:
        tuple: (model_string, hub_height_meters)
    """
    # Handle NaNs or missing types
    if pd.isna(installation_type) or installation_type == "Unknown":
        installation_type = "Onshore"

    if installation_type == "Onshore":
        # Onshore: Trend towards taller towers to capture higher shear
        if start_year <= 2000:
            return "VestasV47_660kW_47", 65.0 
        elif start_year <= 2005:
            return "DOE_GE_1.5MW_77", 80.0
        elif start_year <= 2010:
            return "DOE_GE_1.5MW_77", 90.0 
        elif start_year <= 2015:
            return "2017COE_Market_Average_2.3MW_113", 100.0
        elif start_year <= 2018:
            return "IEA_Reference_3.4MW_130", 125.0
        elif start_year <= 2021:
            return "2020ATB_NREL_Reference_4MW_150", 135.0
        else: 
            return "2023NREL_Bespoke_6MW_170", 150.0

    elif installation_type == "Offshore floating":
        if start_year <= 2020:
            return "IEA_Reference_6MW_100", 110.0
        else:
            return "DTU_Reference_v1_10MW_178", 130.0

    else:  
        if start_year <= 2005:
            return "NREL_Reference_5MW_126", 70.0 # Early offshore (Bonus/Siemens)
        elif start_year <= 2010:
            return "NREL_Reference_5MW_126", 90.0
        elif start_year <= 2015:
            return "LEANWIND_Reference_8MW_164", 100.0 # Siemens 3.6/4.0 era
        elif start_year <= 2019:
            return "DTU_Reference_v1_10MW_178", 110.0 # MHI Vestas 8MW era
        else: 
            return "IEA_Reference_15MW_240", 140.0

def estimate_wind_power(lat, lon, capacity, startyear, installation_type, xrds, 
                        y_idx, x_idx, wts_smoothing=False, power_smoothing=True, 
                        spatial_interpolation=False, wake_loss_factor=None, 
                        single_turb_curve = False, enforce_start_year=False, verbose=True): 

    
    try:
        turbs = Turbines()
        turbine_model, mapped_hub_height = map_turbine_model(startyear, installation_type)
        specs = turbs.specs(turbine_model)
        hub_height = mapped_hub_height if mapped_hub_height else specs['hub_height']
        rated_power_kw = specs['rated_power']

        #Find the index of the height level closest to the hub_height
        available_heights = xrds['heightAboveGround'].values

        ref_height = 100.0
        ref_height_idx = np.abs(available_heights - ref_height).argmin()
        if spatial_interpolation == True:
            wind_ts_ref = interpolate_idw(xrds, lat, lon, 'ws', y_idx, x_idx, ref_height_idx=ref_height_idx, neighbors=4).values
        else:
            wind_ts_ref = xrds['ws'].isel(heightAboveGround=ref_height_idx, y=y_idx, x=x_idx).values

        if installation_type == 'offshore' or installation_type == 'unknown':
            alpha = 1/7 
        else:
            alpha = 0.22
            
        wind_ts_hub_height = wind_ts_ref * (hub_height / ref_height)**alpha
        wind_ts_series = pd.Series(wind_ts_hub_height)
        
        if wts_smoothing:
            wind_ts = wind_ts_series.rolling(window=3, center=True, min_periods=1).mean().values
        else:   
            wind_ts = wind_ts_series.values
        num_turbines = capacity / (rated_power_kw / 1000)

        if single_turb_curve:
            power_curve = turbs.table(turbine_model)
            single_turbine_kw = np.interp(
                wind_ts, 
                power_curve['wind_speed_ms'], 
                power_curve['power_kw'],
                left=0, right=rated_power_kw
            )
            total_farm_power_mw = (single_turbine_kw * num_turbines) / 1e3
        else:
            farm_power_curve = generate_farm_power_curve(turbine_model, num_turbines)
            total_farm_power_mw = np.interp(
                wind_ts, 
                farm_power_curve['wind_speed_ms'], 
                farm_power_curve['power_kw'] / 1e3
        )
        if power_smoothing:
            total_farm_power_mw = pd.Series(total_farm_power_mw).rolling(
                window=6, center=True, min_periods=1
            ).mean().values
            
        if wake_loss_factor is not None:
            total_farm_power_mw *= wake_loss_factor

        return total_farm_power_mw
    except Exception as e:
        if verbose:
            print(f"Could not process farm at ({lat}, {lon}). Error: {e}")
        return None

def generate_farm_power_curve(turb_name, n_turbines):
    """
    Generates a smoothed, farm-level power curve from a single-turbine curve.

    This function convolves the single-turbine power curve with a Gaussian
    distribution of wind speeds to represent the spatial variation of wind
    across a large farm. This results in a smoother power curve that is
    more representative of a wind farm's aggregate output.

    Args:
        turbine_power_curve (dict): A dictionary with 'wind_speed_ms' and 'power_kw' keys.
        n_turbines (int): The number of turbines in the farm.
        wind_speed_std_dev (float): The standard deviation of the wind speed
                                    across the farm. This controls the amount
                                    of "smearing" or smoothing. A larger value
                                    results in a smoother curve. Defaults to 1.5 m/s.

    Returns:
        dict: A new power curve dictionary with 'wind_speed_ms' and 'power_kw' for the entire farm.
    """
    turbs = Turbines()
    specs = turbs.specs(turb_name)

    min_ws = 0.0
    max_ws = 40.0
    ws_step = 0.25
    farm_wind_speeds = np.arange(min_ws, max_ws + ws_step, ws_step)
    farm_power = np.zeros_like(farm_wind_speeds)
    power_curve = turbs.table(turb_name)

    single_turbine_ws = power_curve['wind_speed_ms']
    single_turbine_power = power_curve['power_kw']
    #Interpolation
    interp_ws = np.linspace(min(single_turbine_ws), max(single_turbine_ws), 500)
    interp_power = np.interp(interp_ws, single_turbine_ws, single_turbine_power)

    for i, ws_mean in enumerate(farm_wind_speeds):
        std_dev = 0.6 + 0.1 * farm_wind_speeds[i] #Increasing std dev with wind speed

        wind_dist = norm(loc=ws_mean, scale=std_dev)

        bin_width = interp_ws[1] - interp_ws[0]
        probabilities = wind_dist.pdf(interp_ws) * bin_width

        expected_power_single_turbine = np.sum(interp_power * probabilities)
        farm_power[i] = expected_power_single_turbine * n_turbines

    return {'wind_speed_ms': farm_wind_speeds, 'power_kw': farm_power}

def interpolate_idw(xrds, lat, lon, var, y_idx, x_idx, ref_height_idx, neighbors=4):
    """
    Performs Inverse Distance Weighting (IDW) interpolation for a variable 
    at a specific lat/lon using a window around the nearest grid point.
    """
    # Define a search window around the nearest point (3x3 covers the 4 closest in a regular grid)
    y_min = max(0, y_idx - 1)
    y_max = min(len(xrds['y']), y_idx + 2)
    x_min = max(0, x_idx - 1)
    x_max = min(len(xrds['x']), x_idx + 2)

    # Extract coordinates of the window
    sub_lats = xrds['latitude'].isel(y=slice(y_min, y_max), x=slice(x_min, x_max)).values
    sub_lons = xrds['longitude'].isel(y=slice(y_min, y_max), x=slice(x_min, x_max)).values

    # Calculate squared distances
    dists_sq = (sub_lats - lat)**2 + (sub_lons - lon)**2
    flat_dists = dists_sq.flatten()

    # Find k nearest neighbors
    k = min(neighbors, len(flat_dists))
    idx_nearest = np.argsort(flat_dists)[:k]
    
    # Calculate weights
    dists_nearest = flat_dists[idx_nearest]
    weights = 1 / (dists_nearest + 1e-8) # Avoid division by zero
    weights /= weights.sum()

    # Interpolate
    interpolated_series = 0
    window_shape = sub_lats.shape
    
    for i, flat_idx in enumerate(idx_nearest):
        wy, wx = np.unravel_index(flat_idx, window_shape)
        # Map back to global indices
        gy = y_min + wy
        gx = x_min + wx
        if ref_height_idx is not None:
            interpolated_series += xrds[var].isel(heightAboveGround=ref_height_idx, y=gy, x=gx) * weights[i]
        else:
            interpolated_series += xrds[var].isel(y=gy, x=gx) * weights[i]

    return interpolated_series
