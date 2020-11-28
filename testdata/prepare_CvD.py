""" Prepare CvD models for fitting. """
import os

import numpy as np
from astropy.table import Table, vstack
from astropy.io import fits
import astropy.constants as const
from ppxf import ppxf_util
from spectres import spectres
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import context
from paintbox.utils.broad2res import broad2res

def prepare_VCJ17(data_dir, wave, output, overwrite=False, obsres=100,
                  wextra=50):
    """ Prepare templates for SSP models from Villaume et al. (2017).

        Parameters
    ----------
    data_dir: str
        Path to the SSP models.
    wave: np.array
        Wavelength dispersion.
    output: str
        Name of the output file (a multi-extension FITS file)
    overwrite: bool (optional)
        Overwrite the output files if they already exist.
    obsres: float
        Observed resolution (sigma) in km/s
    wextra: float
        Additional wavelength used at both sides of the data to avoid border
        issues in the data processing.
    """
    if os.path.exists(output) and not overwrite:
        return
    filenames = sorted(os.listdir(data_dir))
    nimf = 16
    imfs = 0.5 + np.arange(nimf) / 5
    x2s, x1s=  np.stack(np.meshgrid(imfs, imfs)).reshape(2, -1)
    ssps, params = [], []
    velscale = obsres / 5
    for fname in tqdm(filenames, desc="Processing SSP files"):
        T = float(fname.split("_")[3][1:])
        Z = float(fname.split("_")[4][1:-8].replace("p", "+").replace(
                    "m", "-"))
        data = np.loadtxt(os.path.join(data_dir, fname))
        w = data[:,0]
        specs = data[:,1:].T
        ########################################################################
        # Trim arrays according to wavelength
        idx = np.where((w >= wave.min() - wextra) &
                       (w <= wave.max() + wextra))[0]
        w = w[idx]
        specs = specs[:, idx]
        # Convolve models to match observed resolution
        if obsres > 100:
            neww = np.exp(ppxf_util.log_rebin([w[0], w[-1]], np.zeros(10),
                                velscale=velscale)[1])
            specs = spectres(neww, w, specs, fill=0, verbose=False).T
            sigma_diff = np.sqrt(obsres**2 - 100**2) / velscale
            w = neww
            specs = gaussian_filter(specs, [sigma_diff, 0])
        specs = spectres(wave, w, specs.T, verbose=False, fill=0.)
        ssps.append(specs)
        for i, (x1, x2) in enumerate(zip(x1s, x2s)):
            params.append(Table([[Z], [T], [x1], [x2]],
                                names=["Z", "Age", "x1", "x2"]))
    ssps = np.vstack(ssps)
    params = vstack(params)
    hdu1 = fits.PrimaryHDU(ssps)
    hdu1.header["EXTNAME"] = "SSPS"
    params = Table(params)
    hdu2 = fits.BinTableHDU(params)
    hdu2.header["EXTNAME"] = "PARAMS"
    # Making wavelength array
    hdu3 = fits.BinTableHDU(Table([wave], names=["wave"]))
    hdu3.header["EXTNAME"] = "WAVE"
    hdulist = fits.HDUList([hdu1, hdu2, hdu3])
    hdulist.writeto(output, overwrite=True)
    return

def prepare_response_functions(data_dir, wave, outprefix, redo=False):
    """ Prepare response functions from CvD models.

    Parameters
    ----------
    data_dir: str
        Path to the response function files
    wave: np.array
        Wavelength dispersion.
    outprefix: str
        First part of the name of the response function output files. The
        response functions are stored in different files for different
        elements, named "{}_{}.fits".format(outprefix, element).
    redo: bool (optional)
        Overwrite output.

    """
    specs = sorted(os.listdir(data_dir))
    # Read one spectrum to get name of columns
    with open(os.path.join(data_dir, specs[0])) as f:
        header = f.readline().replace("#", "")
    fields = [_.strip() for _ in header.split(",")]
    fields[fields.index("C+")] = "C+0.15"
    fields[fields.index("C-")] = "C-0.15"
    fields[fields.index("T+")] = "T+50"
    fields[fields.index("T-")] = "T-50"
    fields = ["{}0.3".format(_) if _.endswith("+") else _ for _ in fields ]
    fields = ["{}0.3".format(_) if _.endswith("-") else _ for _ in fields]
    elements = set([_.split("+")[0].split("-")[0] for _ in fields if
                    any(c in _ for c in ["+", "-"])])
    signal = ["+", "-"]
    for element in tqdm(elements, desc="Preparing response functions"):
        output = "{}_{}.fits".format(outprefix, element.replace("/", ""))
        if os.path.exists(output) and not redo:
            continue
        params = []
        rfs = []
        for spec in specs:
            T = float(spec.split("_")[2][1:])
            Z = float(spec.split("_")[3].split(".abun")[0][1:].replace(
                      "p", "+").replace("m", "-"))
            data = np.loadtxt(os.path.join(data_dir, spec))
            w = data[:,0]
            fsun = data[:,1]
            # Adding solar response
            p = Table([[Z], [T], [0.]], names=["Z", "Age", element])
            rf = np.ones(len(wave))
            rfs.append(rf)
            params.append(p)
            # Adding non-solar responses
            for sign in signal:
                name = "{}{}".format(element, sign)
                cols = [(i,f) for i, f in enumerate(fields) if f.startswith(
                    name)]
                for i, col in cols:
                    val = float("{}1".format(sign)) * float(col.split(sign)[1])
                    t = Table([[Z], [T], [val]], names=["Z", "Age", element])
                    params.append(t)
                    rf = data[:, i] / fsun
                    newrf= spectres(wave, w, rf)
                    rfs.append(newrf)
        rfs = np.array(rfs)
        params = vstack(params)
        hdu1 = fits.PrimaryHDU(rfs)
        hdu1.header["EXTNAME"] = "SSPS"
        params = Table(params)
        hdu2 = fits.BinTableHDU(params)
        hdu2.header["EXTNAME"] = "PARAMS"
        # Making wavelength array
        hdu3 = fits.BinTableHDU(Table([wave], names=["wave"]))
        hdu3.header["EXTNAME"] = "WAVE"
        hdulist = fits.HDUList([hdu1, hdu2, hdu3])
        hdulist.writeto(output, overwrite=True)

def prepare_templates_testdata(zmax=0.05):
    outdir = os.path.join(context.home_dir, "templates")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # Read testdata processed file to obtain wavelength ranges.
    filename = os.path.join(context.home_dir, "data/testdata/NGC7144_spec.fits")
    # Directory where models are stored
    models_dir = "/home/kadu/Dropbox/SPINS/CvD18/"
    ssps_dir = os.path.join(models_dir, "VCJ_v8")
    # Loading the wavelength dispersion from one of the models
    wave = np.loadtxt(os.path.join(ssps_dir, os.listdir(ssps_dir)[0]),
                      usecols=(0,))
    target_res = np.array([200, 100]) # Rounding up the ideal resolution
    velscale = np.ceil(target_res / 2.5) # Same as input spectrum
    wrange = ["blue", "red"]
    for i, ext in enumerate([1,2]):
        table = Table.read(filename, hdu=ext)
        w1data, w2data = table["wave"].min(), table["wave"].max()
        w1 = (w1data / (1 + zmax) / 100).astype(int) * 100
        w2 = (w2data / 100).astype(int) * 100 + 100
        # Using ppxf to obtain a logarithmic-scaled dispersion
        outwave = np.exp(ppxf_util.log_rebin([w1, w2], np.zeros(10),
                                      velscale=velscale[i])[1])
        # Defining where the models should be stored
        output = os.path.join(outdir, "VCJ17_varydoublex_imacs_{"
                                      "}.fits".format(wrange[i]))

        prepare_VCJ17(ssps_dir, outwave, output, obsres=target_res[i])
        input(404)
        # Preparing response functions
        rfs_dir = os.path.join(models_dir, "RFN_v3")
        outprefix = os.path.join(outdir, "C18_imacs_{}_rfs".format(wrange[i]))
        prepare_response_functions(rfs_dir, wave, outprefix)

if __name__ == "__main__":
    prepare_templates_testdata()
