""" Setup paintbox for running on example spectrum. """
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.table import Table
import astropy.constants as const
from ppxf import ppxf_util
from spectres import spectres

from paintbox.utils.broad2res import broad2res

import context

def prepare_spectrum(spec_file, outfile, overwrite=False):
    """ Preparing the spectrum of a single galaxy for the fitting. """
    if os.path.exists(outfile) and not overwrite:
        return
    wave, flux, fluxerr, mask, res_kms = np.loadtxt(spec_file, unpack=True)
    mask = mask.astype(np.bool).astype(np.int)
    # Interpolating flux / fluxerr
    idx = np.where(mask > 0)[0]
    f = interp1d(wave[idx], flux[idx], fill_value="extrapolate")
    flux = f(wave)
    ferr = interp1d(wave[idx], fluxerr[idx], fill_value="extrapolate")
    fluxerr = ferr(wave)
    # Calculating resolution in FWHM
    c = const.c.to("km/s").value
    fwhm = res_kms / c * wave * 2.335
    # Splitting the data to work with different resolutions

    wave_ranges = [[4200, 6680], [8200, 8900]]
    idxs = [np.where((wave >= w1) & (wave < w2))[0] for w1,w2 in wave_ranges]
    waves = [wave[idx] for idx in idxs]
    fluxes = [flux[idx] for idx in idxs]
    fluxerrs = [fluxerr[idx] for idx in idxs]
    masks = [mask[idx] for idx in idxs]
    res_kmss = [res_kms[idx] for idx in idxs]
    fwhms = [fwhm[idx] for idx in idxs]
    # Homogeneize the resolution
    target_res = np.array([200, 100]) # Rounding up the ideal resolution
    velscale = 100
    names = ["wave", "flux", "fluxerr", "mask"]
    hdulist = [fits.PrimaryHDU()]
    for i in range(len(wave_ranges)):
        target_fwhm = target_res[i] / c * waves[i] * 2.355
        flux, fluxerr = broad2res(waves[i], fluxes[i], fwhms[i], target_fwhm,
                                  fluxerr=fluxerrs[i])
        # Using ppxf to obtain a logarithmic-scaled dispersion
        logwave = ppxf_util.log_rebin([waves[i][0], waves[i][-1]], flux,
                                      velscale=velscale)[1]
        newwave = np.exp(logwave)
        # Resampling data
        newflux, newfluxerr = spectres(newwave, waves[i], flux,
                                       spec_errs=fluxerr)
        newmask = spectres(newwave, waves[i], masks[i]).astype(np.int).astype(
                                                                        np.bool)
        obsmask = -1 * (newmask.astype(np.int) - 1)
        table = Table([newwave, newflux, newfluxerr, obsmask], names=names)
        hdu = fits.BinTableHDU(table)
        hdulist.append(hdu)
    hdulist = fits.HDUList(hdulist)
    hdulist.writeto(outfile, overwrite=True)
    return

if __name__ == "__main__":
    wdir = os.path.join(context.home_dir, "data/testdata")
    spec_file = os.path.join(wdir, "NGC7144_1_1arc_bg_noconv.txt")
    outfile = os.path.join(wdir, "NGC7144_spec.fits")
    prepare_spectrum(spec_file, outfile, overwrite=True)