import numpy as np
import pandas as pd
from .turbines import Turbines

def estimate_wind_power(country, lat, lon, capacity, startyear, prod_year, status, installation_type, xrds, 
                        y_idx, x_idx, wts_smoothing=False, power_smoothing=True, 
                        spatial_interpolation=False, wake_loss_factor=None, 
                        single_turb_curve = False, enforce_start_year=False, verbose=True): 
    
    if status not in operating_farms(country, "wind"):
        return None
    if enforce_start_year and (isinstance(startyear, (int, float)) and startyear > prod_year):
        return None
    
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
