import os

import pandas as pd
import plotly.graph_objects as go


def plot(workdir, obs_data_dir, model_id) -> go.Figure:
    files = [os.path.join(workdir, 'calibrated_simulated_flow.nc'), ]
    variables = ('flow_sim', 'flow_bc')
    dim_order = ('time', 'model_id')
    a = grids.TimeSeries(files, variables, dim_order)
    ts = a.point(None, model_id)
    obs = pd.read_csv(os.path.join(obs_data_dir, ))
    a = go.Figure(
        [
            go.Scatter(
                name='Original Simulation',
                x=ts.index,
                y=ts['flow_sim']
            ),
            go.Scatter(
                name='Adjusted Simulation',
                x=ts.index,
                y=ts['flow_bc']
            ),
            go.Scatter(
                name='Adjusted Simulation',
                x=ts.index,
                y=ts['flow_bc']
            )
        ]
    )
    a.show()
    return


def plot_results(sim, obs, bc, bcp, title):
    sim_dates = sim.index.tolist()
    scatters = [
        go.Scatter(
            name='Propagated Corrected (Experimental)',
            x=bcp.index.tolist(),
            y=bcp['Propagated Corrected Streamflow'].values.flatten(),
        ),
        go.Scatter(
            name='Bias Corrected (Jorges Method)',
            x=sim_dates,
            y=bc.values.flatten(),
        ),
        go.Scatter(
            name='Simulated (ERA 5)',
            x=sim_dates,
            y=sim.values.flatten(),
        ),
        go.Scatter(
            name='Observed',
            x=obs.index.tolist(),
            y=obs.values.flatten(),
        ),
        go.Scatter(
            name='Percentile',
            x=sim_dates,
            y=bcp['Percentile'].values.flatten(),
        ),
        go.Scatter(
            name='Scalar',
            x=sim_dates,
            y=bcp['Scalars'].values.flatten(),
        ),
    ]
    a = go.Figure(scatters, layout={'title': title})
    return a
