import pandas as pd
import numpy as np
import xarray as xr


def compute_flow_duration_curve(hydro: list or np.array, prob_steps: int = 400, exceedence: bool = True,
                                col_name: str = 'Flow'):
    percentiles = [round((1 / prob_steps) * i * 100, 5) for i in range(prob_steps + 1)]
    flows = np.nanpercentile(hydro, percentiles)
    if exceedence:
        percentiles.reverse()
    return pd.DataFrame(flows, columns=[col_name, ], index=percentiles)


a = pd.read_csv('data_0_inputs/magdalena_table.csv')
a = a[a['order_'] > 1]
a = sorted(a['COMID'].tolist())

hist_nc = xr.open_dataset('/Users/riley/spatialdata/bias_correction_test_data/south_america_era5_qout.nc')


# start dataframes for the flow duration curve (fdc) and the monthly averages (ma) using the first comid in the list
print('creating first dataframes')
id1 = a.pop(0)
data = hist_nc.sel(rivid=id1).Qout.to_dataframe()['Qout']
fdc_df = compute_flow_duration_curve(data.tolist(), col_name=id1)
ma_df = data.groupby(data.index.strftime('%m')).mean().to_frame(name=id1)

# for each remaining comid in the list, merge the fdc and ma with the previously created df's'
print('appending more comids to initial dataframe')
for comid in a:
    data = hist_nc.sel(rivid=comid).Qout.to_dataframe()['Qout']
    fdc_df = fdc_df.merge(compute_flow_duration_curve(data.tolist(), col_name=comid),
                          how='outer', left_index=True, right_index=True)
    ma_df = ma_df.merge(data.groupby(data.index.strftime('%m')).mean().to_frame(name=comid),
                        how='outer', left_index=True, right_index=True)

# mean_annual_flow = ma_df.mean()
fdc_df.to_csv('data_1_historical_csv/simulated_fdc.csv')
# fdc_df.div(mean_annual_flow).to_csv('data_1_historical_csv/simulated_fdc_normalized.csv')
ma_df.to_csv('data_1_historical_csv/simulated_monavg.csv')
# ma_df.div(mean_annual_flow).to_csv('data_1_historical_csv/simulated_monavg_normalized.csv')
