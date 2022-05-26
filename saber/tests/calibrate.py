from io import StringIO

import geoglows
import pandas as pd
import requests


def collect_data(start_id, start_ideam_id, downstream_id, downstream_ideam_id):
    # Upstream simulated flow
    start = geoglows.streamflow.historic_simulation(start_id)
    # Upstream observed flow
    start_ideam = get_ideam_flow(start_ideam_id)
    start_ideam.dropna(inplace=True)

    # Downstream simulated flow
    downstream = geoglows.streamflow.historic_simulation(downstream_id)
    # Downstream observed flow
    downstream_ideam = get_ideam_flow(downstream_ideam_id)
    downstream_ideam.dropna(inplace=True)
    # Downstream bias corrected flow (for comparison to the data_4_assign_propagation method
    downstream_bc = geoglows.bias.correct_historical(downstream, downstream_ideam)

    # Export all as csv
    start.to_csv('start_flow.csv')
    start_ideam.to_csv('start_ideam_flow.csv')
    downstream.to_csv('downstream_flow.csv')
    downstream_ideam.to_csv('downstream_ideam_flow.csv')
    downstream_bc.to_csv('downstream_bc_flow.csv')
    return


def get_ideam_flow(id):
    # get the gauged data
    url = f'https://www.hydroshare.org/resource/d222676fbd984a81911761ca1ba936bf/' \
          f'data/contents/Discharge_Data/{id}.csv'
    df = pd.read_csv(StringIO(requests.get(url).text), index_col=0)
    df.index = pd.to_datetime(df.index).tz_localize('UTC')
    return df

# collect_data(9012999, 22057070, 9012650, 22057010)
# collect_data(9017261, 32037030, 9015333, 32097010)  # really long range down stream
# collect_data(9009660, 21237020, 9007292, 23097040)  # large river
# collect_data(9007292, 23097040, 9009660, 21237020)  # large river backwards (going upstream)

# Read all as csv
start_flow = pd.read_csv('data_4_assign_propagation/start_flow.csv', index_col=0)
start_ideam_flow = pd.read_csv('data_4_assign_propagation/start_ideam_flow.csv', index_col=0)
downstream_flow = pd.read_csv('data_4_assign_propagation/downstream_flow.csv', index_col=0)
downstream_ideam_flow = pd.read_csv('data_4_assign_propagation/downstream_ideam_flow.csv', index_col=0)
downstream_bc_flow = pd.read_csv('data_4_assign_propagation/downstream_bc_flow.csv', index_col=0)
start_flow.index = pd.to_datetime(start_flow.index)
start_ideam_flow.index = pd.to_datetime(start_ideam_flow.index)
downstream_flow.index = pd.to_datetime(downstream_flow.index)
downstream_ideam_flow.index = pd.to_datetime(downstream_ideam_flow.index)
downstream_bc_flow.index = pd.to_datetime(downstream_bc_flow.index)

downstream_prop_correct = rbc_calibrate(start_flow, start_ideam_flow, downstream_flow,
                                        fit_gumbel=True, gumbel_range=(25, 75))
plot_results(downstream_flow, downstream_ideam_flow, downstream_bc_flow, downstream_prop_correct,
             f'Correct Monthly - Force Gumbel Distribution')
del downstream_prop_correct['Scalars'], downstream_prop_correct['Percentile']
statistics_tables(downstream_prop_correct, downstream_flow, downstream_ideam_flow).to_csv(
    'data_4_assign_propagation/stats_test.csv')
