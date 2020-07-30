import requests
import pandas as pd
from io import StringIO
import numpy as np
import glob
import os


def compute_flow_duration_curve(hydro: list or np.array, prob_steps: int = 400, exceedence: bool = True,
                                col_name: str = 'Flow'):
    percentiles = [round((1 / prob_steps) * i * 100, 5) for i in range(prob_steps + 1)]
    flows = np.nanpercentile(hydro, percentiles)
    if exceedence:
        percentiles.reverse()
    return pd.DataFrame(flows, columns=[col_name, ], index=percentiles)


# ideam_ids = pd.read_csv('data_0_inputs/magdalena_stations_table.csv')['ID'].tolist()
#
# for id in ideam_ids:
#       # get the gauged data
#       url = f'https://www.hydroshare.org/resource/d222676fbd984a81911761ca1ba936bf/' \
#             f'data/contents/Discharge_Data/{id}.csv'
#       df = pd.read_csv(StringIO(requests.get(url).text), index_col=0)
#       df.index = pd.to_datetime(df.index).tz_localize('UTC')
#       df.to_csv(f'data_4_observations/ideam_raw_csvs/{id}.csv')

csvs = sorted(glob.glob('data_0_inputs/ideam_raw_csvs/*.csv'))

# start dataframes for the flow duration curve (fdc) and the monthly averages (ma) using the first comid in the list
print('creating first dataframes')
first_csv = csvs.pop(0)
id = os.path.splitext(os.path.basename(first_csv))[0]
data = pd.read_csv(first_csv, index_col=0, parse_dates=True)
data.rename(columns={data.columns[0]: id}, inplace=True)
fdc_df = compute_flow_duration_curve(data.values.tolist(), col_name=id)
ma_df = data.groupby(data.index.strftime('%m')).mean()

# for each remaining comid in the list, merge the fdc and ma with the previously created df's'
print('appending more comids to initial dataframe')
for i, csv in enumerate(csvs):
    id = os.path.splitext(os.path.basename(csv))[0]
    data = pd.read_csv(csv, index_col=0, parse_dates=True)
    data.rename(columns={data.columns[0]: id}, inplace=True)
    fdc_df = fdc_df.merge(compute_flow_duration_curve(data.values.tolist(), col_name=id),
                          how='outer', left_index=True, right_index=True)
    ma_df = ma_df.merge(data.groupby(data.index.strftime('%m')).mean(),
                        how='outer', left_index=True, right_index=True)

mean_annual_flow = ma_df.mean()
fdc_df.to_csv('data_1_historical_csv/observed_fdc.csv')
fdc_df.div(mean_annual_flow).to_csv('data_1_historical_csv/observed_fdc_normalized.csv')
ma_df.to_csv('data_1_historical_csv/observed_monavg.csv')
ma_df.div(mean_annual_flow).to_csv('data_1_historical_csv/observed_monavg_normalized.csv')


# need to make sure that there are no invalid observational datasets
import glob
import os
csv_ids = []
for csv in glob.glob('data_0_inputs/ideam_raw_csvs/*.csv'):
    csv_ids.append(os.path.splitext(os.path.basename(csv))[0])
csv_ids = sorted(csv_ids)
print(csv_ids)
print(len(csv_ids))
a = pd.read_csv('data_1_historical_csv/observed_monavg_normalized.csv', index_col=0)
monavg_col_ids = sorted(a.columns.tolist())
print(monavg_col_ids)
print(len(monavg_col_ids))

a = pd.read_csv('data_1_historical_csv/observed_fdc_normalized.csv', index_col=0)
fdc_col_ids = sorted(a.columns.tolist())
print(fdc_col_ids)
print(len(fdc_col_ids))
