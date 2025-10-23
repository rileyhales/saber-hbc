import json
import os
import warnings
from glob import glob
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
from natsort import natsorted
from tqdm import tqdm

# Suppress RankWarning
warnings.simplefilter('ignore', np.exceptions.RankWarning)

qtop_polyfits_dir = '/home/ubuntu/polyfits/qtop_df'
ptoq_polyfits_dir = '/home/ubuntu/polyfits/ptoq_df'
fdc_zarr = '/home/ubuntu/fdc.zarr'
saber_zarr = '/home/ubuntu/saber.zarr'
polyfits_zarr = '/home/ubuntu/polyfits.zarr'
processing_chunk_size = 500_000

os.makedirs(qtop_polyfits_dir, exist_ok=True)
os.makedirs(ptoq_polyfits_dir, exist_ok=True)

intermediate_processing_chunk_size = processing_chunk_size // os.cpu_count()
zarr_chunk_size = 5_000
qtop_degree = 7
ptoq_degree = 7
months = np.arange(1, 13)

fdc_ds = xr.open_zarr(fdc_zarr)
saber_ds = xr.open_zarr(saber_zarr)

ds_attributes = {
    'description': f'Polynomial coefficients for converting raw discharge into transformed discharge',
    'units': 'p_exceed is the exceedance probability as an integer percentage (%), Q is daily average discharge in units of cubic meters per second',
    'QtoP_transformation': 'p_values = np.exp(np.poly1d(coefficients)(q_values)) - 1, where poly is the polynomial defined by coefficients in QtoP for the month and river_id.',
    'PtoQ_transformation': 'q_values = np.exp(np.poly1d(coefficients)(p_values)) - 1, where poly is the polynomial defined by coefficients in PtoQ for the month and river_id.',
    'note': 'Clip Q to minmax before qtop_df transformation and clip p_exceed to [0, 100] before ptoq_df transformation.',
    'qtop_polynomial_degree': qtop_degree,
    'ptoq_polynomial_degree': ptoq_degree,
}


def ptoq_polyfit(river_id: str) -> xr.Dataset:
    x_var = 'p_exceed'
    y_var = 'fdc_transformed'

    df = xr.open_dataset(saber_zarr).sel(river_id=river_id)[y_var].to_dataframe().reset_index()[['month', x_var, y_var]]
    monthly_values = {}
    for month in months:
        # fit a polynomial of log(sfdc) (y) onto p_exceed (x)
        monthly = df.loc[df['month'] == month, [x_var, y_var]]
        x = monthly[x_var].values
        y = monthly[y_var].values
        coefficients = np.polyfit(x, np.log(y + 1), deg=ptoq_degree)
        monthly_values[month] = coefficients
        # y_fit = np.exp(np.poly1d(coefficients)(x)) - 1
        # generate_plot(x, y, y_fit, month, river_id, plot_type='sfdc')

    return (
        xr
        .Dataset(
            data_vars={
                'PtoQ': (('river_id', 'month', 'ptoq_exponent_degree'), np.array([monthly_values[m] for m in months]).reshape(1, 12, -1)),
            },
            coords={
                'river_id': [river_id, ],
                'month': months,
                'ptoq_exponent_degree': np.arange(0, ptoq_degree + 1)[::-1],
            },
        )
    )


def qtop_polyfit(river_id: int) -> xr.Dataset:
    # for each month, there will be a degree + 1 coefficients and we should also store the min and max of daily_monthly for that month so that
    # we can clip values outside of that range when using the polynomial for transformations later
    # normally we have exceedance probability (x) and want to get daily_monthly (y)
    # the regression should be inverted so we can transform a given daily_monthly value into an exceedance probability
    # if there is insufficient variance in the dataset, fit a constant value of 50% exceedance
    degree = 7
    try:
        df = (
            fdc_ds
            .daily_monthly
            .sel(river_id=river_id)
            .to_dataframe()
            [['daily_monthly']]
            .reset_index()
        )
        monthly_values = {}
        for month in range(1, 13):
            monthly = df.loc[df['month'] == month, ['daily_monthly', 'p_exceed']]
            x = monthly['daily_monthly'].values
            y = monthly['p_exceed'].values
            x_max = x[0]
            x_min = x[-1]
            # if the std is less than 5 or all values are less than 1 or is entirely nan
            if np.isnan(x).all() or np.nanstd(x) < 5 or np.all(x < 1):
                coefficients = np.array([0] * degree + [np.log(50), ])
            else:
                coefficients = np.polyfit(x, np.log(y + 1), deg=degree)

            monthly_values[month] = {
                'coefficients': coefficients,
                'x_min': x_min,
                'x_max': x_max,
            }

            # y_fit = np.exp(np.poly1d(coefficients)(x)) - 1
            # generate_plot(x, y, y_fit, month, river_id, plot_type='fdc')

        # now write to a netcdf with dimensions of river_id, month, coefficient
        coeffs_array = np.array([monthly_values[m]['coefficients'] for m in range(1, 13)]).reshape(1, 12, -1)
        minmaxes = np.array([[monthly_values[m]['x_min'], monthly_values[m]['x_max']] for m in range(1, 13)]).reshape(1, 12, 2)
        return (
            xr
            .Dataset(
                data_vars={
                    'QtoP': (('river_id', 'month', 'qtop_exponent_degree'), coeffs_array),
                    'Qrange': (('river_id', 'month', 'minmax'), minmaxes),
                },
                coords={
                    'river_id': [river_id, ],
                    'month': np.arange(1, 13),
                    'qtop_exponent_degree': np.arange(0, degree + 1)[::-1],
                    'minmax': ['min', 'max'],
                },
            )
        )

    except Exception as e:
        print(f'Error processing river_id {river_id}: {e}')
        return river_id


def generate_plot(x, y, y_fit, month, river_id, plot_type: str = None) -> None:
    if plot_type == 'sfdc':
        plot_title = f'SFDC Fit for Month {month}, River ID {river_id}'
        y_label = 'SFDC value'
    elif plot_type == 'fdc':
        plot_title = f'FDC Fit for Month {month}, River ID {river_id}'
        y_label = 'Q (daily_monthly)'
    else:
        plot_title = 'Unknown plot type'
        y_label = 'Value'

    # make a plot of the graphs and save it to a file
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, label='Original', marker='o', color='blue')
    ax.plot(x, y_fit, label='Computed', marker='x', color='red')
    ax.set_xlabel('Exceedance Probability (%)')
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    ax.legend()
    plt.grid()
    plt.show()
    return


def partial_concat(datasets: list) -> xr.Dataset:
    return xr.concat(datasets, dim='river_id')


if __name__ == '__main__':
    # todo: apply the sfdc normalization to the sfdcs
    """
    the sequence of steps should be:
    1. generate sfdcs at all gauge/river pairs
    2. generate assignments table
    3. densify the sfdcs from gauged rivers to all rivers (e.g. ~30k points to 6.8M points)
    4. fit regression equations
        1. Q -> p_exceed (fdc)
        2. p_exceed -> Q (fdc transformed)
    5. store regression coefficients for later use in bias correction
    6. apply the regression equations to bias correct daily flows
    
    # possible alternative option if the fit is good and generalizable shape:
    4. fit Q to Q bias corrected for each month
        1. Join FDC (Q vs exceedance probability) with SFDC (SAF vs exceedance probability) along exceedance probability axis.
        2. Calculate Q bias corrected by multiplying Q by SAF
        3. Fit a polynomial to the relationship between log(Q) (x) and log(Q bias corrected) (y) for each month.
        4. Store polynomial coefficients for each month for later use.
    """
    # fdc_polyfit(820173701)
    # fdc_polyfit(710431167)
    # ptoq_polyfit(710431167)
    with Pool() as p:
        for ds, save_dir, func in (
                (fdc_ds, qtop_polyfits_dir, qtop_polyfit),
                (saber_ds, ptoq_polyfits_dir, ptoq_polyfit),
        ):
            river_ids = ds.river_id.values.flatten()
            for i in range(0, len(river_ids)):
                save_path = os.path.join(save_dir, f'{i // processing_chunk_size:03d}_stepsize{processing_chunk_size}.nc')
                if os.path.exists(save_path):
                    continue
                river_id_chunk = river_ids[i:i + processing_chunk_size]
                results = list(tqdm(p.imap(func, river_id_chunk), total=len(river_id_chunk)))
                results = [results[i:i + intermediate_processing_chunk_size] for i in range(0, len(results), intermediate_processing_chunk_size)]
                combined = list(p.imap(partial_concat, results))
                combined = xr.concat(combined, dim='river_id')
                combined.to_netcdf(save_path)
                print(f'Saved {save_path}')

    (
        xr
        .open_mfdataset(
            natsorted(glob(os.path.join(qtop_polyfits_dir, '*.nc')))
        )
        .chunk({
            'river_id': zarr_chunk_size,
            'month': -1,
            'minmax': -1,
            'qtop_exponent_degree': -1,
        })
        .assign_attrs(ds_attributes)
        .to_zarr(
            polyfits_zarr,
            mode='w',
            encoding={
                'QtoP': {'compressor': zarr.Blosc(cname='zstd', clevel=5, shuffle=2)},
            }
        )
    )
    # now rewrite the river_id coordinate variable chunks into 1 piece instead of the 1-per-batch that it gets appended in
    with open(os.path.join(polyfits_zarr, 'river_id', '.zattrs'), 'r') as f:
        river_id_zattrs = f.read()
    zgroup = zarr.open(polyfits_zarr, mode='a')
    ids = zgroup['river_id'][:]
    zgroup.create_dataset(
        'river_id',
        data=ids,
        shape=ids.shape,
        dtype=ids.dtype,
        chunks=ids.shape,
        overwrite=True,
    )
    zmetadata_path = os.path.join(polyfits_zarr, '.zmetadata')
    with open(zmetadata_path, 'r') as f:
        zmetadata = json.load(f)
    zmetadata['metadata']["river_id/.zarray"]["chunks"] = zmetadata['metadata']["river_id/.zarray"]["shape"]
    with open(zmetadata_path, 'w') as f:
        json.dump(zmetadata, f, indent=2)
    with open(os.path.join(polyfits_zarr, 'river_id', '.zattrs'), 'w') as f:
        f.write(river_id_zattrs)

    # now do the same thing for the ptoq_df polyfits
    (
        xr
        .open_mfdataset(
            natsorted(glob(os.path.join(ptoq_polyfits_dir, '*.nc')))
        )
        .chunk({
            'river_id': zarr_chunk_size,
            'month': -1,
            'ptoq_exponent_degree': -1,
        })
        .assign_attrs(ds_attributes)
        .to_zarr(
            polyfits_zarr,
            mode='a',
            encoding={
                'PtoQ': {'compressor': zarr.Blosc(cname='zstd', clevel=5, shuffle=2)},
            }
        )
    )
