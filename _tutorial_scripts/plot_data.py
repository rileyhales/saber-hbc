import pandas as pd
import plotly.graph_objects as go

a = pd.read_csv('../old_data/data_extras/order>1/simulated_monavg_normalized.csv', index_col=0)

months = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
scatters = []
for column in a.columns:
    scatters.append(go.Scatter(
        name=column,
        x=months,
        y=a[column].values,
    ))
go.Figure(data=scatters, layout={'title': 'Plot All'}).show()
