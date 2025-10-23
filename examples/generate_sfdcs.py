import json
import os
from multiprocessing import Pool

import numcodecs
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from tqdm import tqdm

import saber

gauge_data_csv_directory = ''
fdc_zarr = './fdc.zarr'
saber_zarr = './saber.zarr'

fdc_ds = xr.open_zarr(fdc_zarr)
sfdcs_dir = './sfdcs'

os.makedirs(sfdcs_dir, exist_ok=True)


def wrapper(args):
    return generate_sfdc(*args)


def generate_sfdc(gauge_id, model_id):
    gauge_file = os.path.join(gauge_data_csv_directory, f'{gauge_id}.csv')
    output_sfdc_file = os.path.join(sfdcs_dir, f'{model_id}.nc')
    try:
        if os.path.exists(gauge_file):
            sim_fdc = (
                fdc_ds
                .sel(river_id=model_id)
                ['daily_monthly']
                .to_dataframe()
                .drop(columns='river_id')
                .reset_index()
                .pivot(index='month', columns='p_exceed', values='daily_monthly')
                .sort_index()
            )
            obs_df = pd.read_csv(gauge_file, index_col=0)
            obs_df.index = pd.to_datetime(obs_df.index)
            obs_df = obs_df[obs_df.iloc[:, 0] >= 0]  # ignore rows with negative values

            sfdcs = []
            for month in range(1, 13):
                month_df = saber.fdc.sfdc(
                    sim_fdc.loc[month],
                    saber.fdc.fdc(obs_df[obs_df.index.month == month].values.flatten())
                )
                # now add a column with the month value and assign it as an index column
                month_df.index.name = 'p_exceed'
                month_df['month'] = month
                month_df = month_df.set_index('month', append=True).reorder_levels(['month', 'p_exceed']).sort_index()
                sfdcs.append(month_df)
            sfdc_df = pd.concat(sfdcs).reorder_levels(['month', 'p_exceed']).reset_index().pivot(index='p_exceed', columns='month', values='scalars')
        else:
            print(f'Gauge file {gauge_file} does not exist. Generating ones.')
            sfdc_df = pd.DataFrame(
                np.ones((101, 12)),
                index=np.linspace(0, 100, num=101).astype(int),
                columns=list(range(1, 13)),
            )
    except Exception as e:
        print(f'unexpected error working on {gauge_id}')
        # traceback.print_exc()
        sfdc_df = pd.DataFrame(
            np.ones((101, 12)),
            index=np.linspace(0, 100, num=101).astype(int),
            columns=list(range(1, 13)),
        )

    # check for nan or inf values
    if np.isnan(sfdc_df.values).any():
        print(f'NaN found in SFDC for gauge {gauge_id} and model {model_id}.')
        # return gauge_id
        sfdc_df = sfdc_df.fillna(1)
    if np.isinf(sfdc_df.values).any():
        print(f'Inf found in SFDC for gauge {gauge_id} and model {model_id}.')
        return gauge_id
    if (sfdc_df.values < 0).any():
        print(f'Negative value found in SFDC for gauge {gauge_id} and model {model_id}.')
        return gauge_id

    (
        xr
        .Dataset(
            {
                'sfdc': (('river_id', 'p_exceed', 'month'), sfdc_df.values[np.newaxis, :, :]),
            },
            coords={
                'river_id': [model_id, ],
                'p_exceed': sfdc_df.index.values,
                'month': sfdc_df.columns.values,
            }
        )
        .to_netcdf(
            output_sfdc_file,
            unlimited_dims=['river_id']
        )
    )


if __name__ == "__main__":
    df = pd.read_parquet('./assign_table-final.parquet')
    df['model_id'] = df['model_id'].astype(int)
    df['asgn_mid'] = df['asgn_mid'].astype(int)
    gauged_rows = df.loc[df['gauge_id'].notna(), ['gauge_id', 'model_id']]

    with Pool() as p:
        errors = list(tqdm(
            p.imap_unordered(wrapper, gauged_rows.values),
            total=gauged_rows.shape[0],
            desc='Generating SFDCs'
        ))
    # for row in tqdm(gauged_rows.values, total=gauged_rows.shape[0], desc='Generating SFDCs'):
    #     error = wrapper(row)

    # add one more with a dummy value pair for model_id -1 and gauge_id '0000000' to ensure that the dataset has at least one entry
    generate_sfdc(gauge_id=False, model_id=-1)

    if os.path.exists('./sfdcs.nc'):
        os.remove('./sfdcs.nc')
    os.system('ncrcat -O ./sfdcs/*.nc ./sfdcs.nc')

    ds = xr.load_dataset('./sfdcs.nc')
    ds.attrs.pop('NCO', None)
    ds.attrs.pop('history', None)
    df = df.sort_values(by='model_id')[['model_id', 'gauge_id', 'asgn_mid', 'asgn_gid']]
    df.loc[df['asgn_gid'].isna(), 'asgn_mid'] = -1

    # now make an sfdc dataset where the river_id is the model_id and the data is aligned correctly.
    # every row should have an entry, not just the ones which were used to generate the sfdcs originally
    compressor = numcodecs.Zstd(level=9)
    batch_size = 5_000
    size = len(df)
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, size)
        asgn_mid_batch = df['asgn_mid'].values[start:end]
        model_id_batch = df['model_id'].values[start:end]
        batch_ds = (
            ds
            .sel(river_id=asgn_mid_batch)
            .assign_coords(river_id=model_id_batch)
            .chunk({
                'river_id': batch_size,
                'p_exceed': 101,
                'month': 12
            })
        )
        if start == 0:
            (
                batch_ds
                .chunk({
                    'river_id': batch_size,
                    'p_exceed': 101,
                    'month': 12
                })
                .to_zarr(
                    saber_zarr,
                    mode='w',
                    consolidated=True,
                    zarr_format=2,
                    encoding={
                        'sfdc': {
                            'compressor': compressor,
                            'dtype': 'float32',
                        }
                    }
                )
            )
        else:
            batch_ds.to_zarr(
                saber_zarr,
                mode='a',
                append_dim='river_id',
                consolidated=True,
                zarr_format=2,
            )

    # now rewrite the river_id coordinate variable chunks into 1 piece instead of the 1-per-batch that it gets appended in
    zgroup = zarr.open(saber_zarr, mode='a')
    ids = zgroup['river_id'][:]
    zgroup.create_dataset(
        'river_id',
        data=ids,
        shape=ids.shape,
        dtype=ids.dtype,
        chunks=ids.shape,
        overwrite=True,
    )
    zmetadata_path = os.path.join(saber_zarr, '.zmetadata')
    with open(zmetadata_path, 'r') as f:
        zmetadata = json.load(f)
    zmetadata['metadata']["river_id/.zarray"]["chunks"] = zmetadata['metadata']["river_id/.zarray"]["shape"]
    with open(zmetadata, 'w') as f:
        json.dump(zmetadata, f)

    # now precompute all corrected fdcs by multiplying sfdcs by fdcs on matching dimensions
    fdc_ds = xr.open_zarr(fdc_zarr)
    sfdc_ds = xr.open_zarr(saber_zarr)
    fdcs = fdc_ds.daily_monthly.transpose('river_id', 'p_exceed', 'month')
    sfdcs = sfdc_ds.sfdc.transpose('river_id', 'p_exceed', 'month').clip(0.1, 10)
    corrected_fdcs = fdcs / sfdcs
    (
        corrected_fdcs
        .to_dataset(name='fdc_transformed')
        .chunk({
            'river_id': 200,
            'p_exceed': 101,
            'month': 12,
        })
        .to_zarr(
            saber_zarr, mode='a', consolidated=True,
            encoding={
                'fdc_transformed': {
                    'chunks': (200, 101, 12), 'compressor': zarr.get_codec({'id': 'zstd', 'level': 9}), 'dtype': 'float32'
                }
            }
        )
    )
