import h5py
import random 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pysteps.visualization import plot_precip_field
import matplotlib.animation as animation
from  matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from IPython.display import HTML

from pincast_verif.io_tools import arr_reconstruct_uint8, dBZ_to_rainrate


def read_nowcast(
    db_path: str,
    method_name : str = None,
    n_sample: int = 0,
    timestamps: list = None,
    leadtimes: list = None,
    unit: str = None,
    ) -> np.ndarray:

    if n_sample == 0 and timestamps is None:
        raise ValueError("either num_sample > 0 or timestamp list must be specified")
    # setup...
    with h5py.File(db_path, 'r') as f:
        all_timestamps = list(f.keys())
        first_ts = all_timestamps[0]
        if n_sample > 0:
            timestamps = random.sample(all_timestamps, k=n_sample)
        if method_name is None:
            method_name = list(f[first_ts].keys())[0]
        if leadtimes is None:
            leadtimes = list(f[f"{first_ts}/{method_name}"].keys())
        else:
            leadtimes = [str(lt) for lt in leadtimes]
        leadtimes.sort(key=float)
        
        arr_shape = f[f"{first_ts}/{method_name}/{leadtimes[0]}/data"].shape
        
        data = np.empty((len(timestamps), len(leadtimes), *arr_shape))
        metadata = pd.DataFrame(columns=["timestamp", "model", "leadtime", "i", "j"])

        for i,ts in enumerate(timestamps):
            for j,lt in enumerate(leadtimes):
                data[i,j] = f[f"{ts}/{method_name}/{lt}/data"]
                metadata.loc[len(metadata)] = [str(ts), method_name, lt, i, j]
        
        if unit == "mm/h":
            data = dBZ_to_rainrate(arr_reconstruct_uint8(data))
        elif unit == "dBZ":
            data = arr_reconstruct_uint8(data)
    
    return data, metadata


def animate(
    data: np.ndarray,
    metadata: pd.DataFrame,
    event_i: int,
    save: bool = False,
    savename: str = None,
    show: bool = True,
    interval=200
    ):

    if save and (savename is None):
        ts = metadata["timestamp"].iloc[event_i]
        model = metadata["model"].iloc[event_i]
        savename = f"{ts}_{model}.mp4"
    fig, ax = plt.subplots()
    n_frames = data.shape[1]
    norm = PowerNorm(gamma=0.25, vmin=0, vmax=160,clip=True)
    im = ax.imshow(data[event_i,0,:,:], norm=norm)

    def init():
        im.set_data(data[event_i,0,:,:])
        return (im,)

    # animation function. This is called sequentially
    def animate(i):
        data_slice = data[event_i, i,:,:]
        im.set_data(data_slice)
        return (im,)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=n_frames, interval=interval, blit=True)

    
    if save:
        anim.save(savename)
    plt.close()
    if show:
        return HTML(anim.to_html5_video())


def plot_overview(
    data: np.ndarray,
    metadata: pd.DataFrame,
    fig_shape: tuple = None,
    save: bool = False,
    unit: str = None,
    ):
    """Plot a broad overview of data events

    Args:
        data (np.ndarray): _description_
        metadata (pd.DataFrame): _description_
        fig_shape (tuple, optional): _description_. Defaults to None.
        save (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if fig_shape is None:
        n_rows = int(np.sqrt(data.shape[0]))
        n_cols = int(data.shape[0] / n_rows) + 1
    else:
        n_rows, n_cols = fig_shape
    assert n_rows > 1 and n_cols > 1 ,"both number of rows and columns must be superior to one"
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(2*n_rows,2*n_cols))
    counter = 0
    for axrow in axes:
        for ax in axrow:
            ax.set_xticks([])
            ax.set_yticks([])
            try:
                if unit is None:
                    ax.imshow(data[counter,0])
                else:
                    plot_precip_field(data[counter,0], units=unit, ax=ax, colorbar=False)
                ax.set_title(counter)
            except IndexError:
                ax.axis("off")
            
            counter +=1
    plt.close()
    return fig

    
def plot_prob_nowcast_mmh(
    observations_mmh,
    prediction_means_mmh,
    prediction_stds_mmh,
    n_stds: int = 2,
    leadtimes: list = [0,2,5,11],
    savefig: bool = False,
    savename: str = None,
    ):

    fig, ax = plt.subplots(nrows=len(leadtimes), ncols=3, figsize=(4*len(leadtimes),6*3))
    for i,lt in enumerate(leadtimes):
        
        divider = make_axes_locatable(ax[i,0])
        cax0 = divider.new_horizontal(size='5%', pad=0.9, pack_start = True)
        fig.add_axes(cax0)
        ax[i,0] = plot_precip_field(observations_mmh[lt], ax=ax[i,0], cax=cax0)
        ax[i,0].set_ylabel(f"+{(lt+1)*5} min")
        ax[i,0].set_title("Observation")

        divider = make_axes_locatable(ax[i,1])
        cax1 = divider.new_horizontal(size='5%', pad=0.9, pack_start = True)
        fig.add_axes(cax1)
        ax[i,1] = plot_precip_field(prediction_means_mmh[lt], ax=ax[i,1], cax=cax1)
        ax[i,1].set_ylabel(f"+{(lt+1)*5} min")
        ax[i,1].set_title("Prediction Mean")

        norm = PowerNorm(gamma=0.25, vmin=0,vmax=160, clip=True)
        im  = ax[i,2].imshow(prediction_stds_mmh[lt]*n_stds, cmap='plasma', norm=norm)
        ax[i,2].axis('off')
        divider = make_axes_locatable(ax[i,2])
        cax2 = divider.new_horizontal(size='5%', pad=0.9, pack_start = True)
        fig.add_axes(cax2)
        cbar = fig.colorbar(im, cax = cax2, orientation = 'vertical')
        cbar.set_ticks([0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0])
        cbar.set_ticklabels([0.0, 1.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0])
        cbar.ax.set_ylabel('Predictive uncertainty [mm/h]', rotation=90)
        ax[i,2].set_title("Predictive Uncertainty")

    if savefig:
        assert savename is not None
        fig.savefig(savename)
    plt.close()
    return fig


def plot_exc_probs(
    exc_probs, 
    threshs, 
    observations = None,
    leadtimes: list = [0,2,5,11],
    savefig: bool = False,
    savename: str = None,
    ):
    
    oinn = observations is not None
    ncols=len(threshs) + oinn
    fig, ax = plt.subplots(nrows=len(leadtimes), ncols=ncols, figsize=(6*ncols,5*len(leadtimes)))

    for i,lt in enumerate(leadtimes):
        if oinn:
            divider = make_axes_locatable(ax[i,0])
            cax0 = divider.new_horizontal(size='5%', pad=0.4, pack_start = False)
            fig.add_axes(cax0)
            ax[i,0] = plot_precip_field(observations[lt], ax=ax[i,0], cax=cax0, map_kwargs={"alpha":0.4})
            ax[i,0].set_ylabel(f"+{(lt+1)*5} min")
            ax[i,0].set_title("Observation")
            
        for j, thr in enumerate(threshs):
            divider = make_axes_locatable(ax[i,oinn+j])
            cax = divider.new_horizontal(size='5%', pad=0.4, pack_start = False)
            fig.add_axes(cax)
            ax[i,j] = plot_precip_field(exc_probs[lt,j], ptype="prob", probthr=thr, ax=ax[i,oinn+j], cax=cax)
            ax[i,oinn+j].set_title("Exceedence probability")

    if savefig:
        assert savename is not None
        fig.savefig(savename)
    plt.close()
    return fig
