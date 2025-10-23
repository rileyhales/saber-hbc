from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

retrospective_zarr = 's3://geoglows-v2/retrospective/daily.zarr'
polyfits_zarr = 's3://geoglows-v2/transformers/polyfits.zarr'

# select the river_id to convert
river_id = 770323095

t1 = datetime.now()

# dataframe to transform
q = (
    xr
    .open_zarr(retrospective_zarr, storage_options={'anon': True})
    .sel(river_id=river_id)
    .Q
    .to_dataframe()
    .reset_index()
    .pivot(index='time', columns='river_id', values='Q')
)
months = q.index.month.unique().sort_values().values

polyfits_ds = xr.open_zarr(polyfits_zarr).sel(river_id=river_id)
qrange = (
    polyfits_ds
    [['Qrange']]
    .to_dataframe()
    .drop(columns='river_id')
    .reset_index()
)
qtop_df = (
    polyfits_ds
    [['QtoP']]
    .to_dataframe()
    .drop(columns='river_id')
    .reset_index()
)
ptoq_df = (
    polyfits_ds
    [['PtoQ']]
    .to_dataframe()
    .drop(columns='river_id')
    .reset_index()
)

t2 = datetime.now()

transformed = []
for month in months:
    month_qrange = qrange.loc[qrange['month'] == month, 'Qrange'].values.flatten()
    month_qtop_coefficients = qtop_df.loc[qtop_df['month'] == month, 'QtoP'].values.flatten()
    month_ptoq_coefficients = ptoq_df.loc[ptoq_df['month'] == month, 'PtoQ'].values.flatten()
    qtop = np.poly1d(month_qtop_coefficients)
    ptoq = np.poly1d(month_ptoq_coefficients)

    monthly_df = q.loc[q.index.month == month].clip(month_qrange[0], month_qrange[1])
    monthly_q = monthly_df.values.flatten()
    p = np.exp(qtop(monthly_q)) - 1
    p = np.clip(p, 0, 100)
    q_transformed = np.exp(ptoq(p)) - 1
    q_transformed = pd.DataFrame(
        q_transformed.reshape(monthly_df.shape),
        index=monthly_df.index,
        columns=monthly_df.columns,
    )
    transformed.append(q_transformed)
transformed = pd.concat(transformed).sort_index()

t3 = datetime.now()

transformed = transformed.merge(q, left_index=True, right_index=True, suffixes=('_transformed', '_original'))
transformed.plot(figsize=(12, 6))
plt.title(f'Transformed Q for river_id: {river_id}')
plt.xlabel('Datetime')
plt.ylabel('Q')
plt.show()

t4 = datetime.now()
print((t3 - t2).total_seconds())
print((t4 - t1).total_seconds())
