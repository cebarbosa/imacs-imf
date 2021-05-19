""" Run paintbox in observed data. """
import os
import shutil
import copy
import glob
import platform

import numpy as np
from scipy import stats
import multiprocessing as mp
from astropy.table import Table
import emcee
import paintbox as pb
from paintbox.utils import CvD18, disp2vel

import context

def make_paintbox_model(wave, outname, name="test", porder=45, nssps=1,
                        sigma=100):
    # Directory where you store your CvD models
    base_dir = context.cvd_dir
    # Locationg where pre-processed models will be stored for paintbox
    outdir = os.path.join(context.home_dir, f"templates")
    # Indicating the filenames of the SSP models
    ssps_dir = os.path.join(base_dir, "VCJ_v8")
    ssp_files = glob.glob(os.path.join(ssps_dir, "VCJ*.s100"))
    # Indicating the filenames for the response functions
    rfs_dir = os.path.join(base_dir, "RFN_v3")
    rf_files = glob.glob(os.path.join(rfs_dir, "atlas_ssp*.s100"))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # Defining wavelength for templates
    velscale = sigma / 2
    wmin = wave.min() - 200
    wmax = wave.max() + 50
    twave = disp2vel([wmin, wmax], velscale)
    ssp = CvD18(twave, ssp_files=ssp_files, rf_files=rf_files, sigma=sigma,
                outdir=outdir, outname=outname)
    limits = ssp.limits
    if nssps > 1:
        for i in range(nssps):
            p0 = pb.Polynomial(twave, 0)
            p0.parnames = [f"p{name}_{0}_{i+1}"]
            s = copy.deepcopy(ssp)
            s.parnames = ["{}_{}".format(_, i+1) for _ in s.parnames]
            if i == 0:
                pop = p0 * s
            else:
                pop += (p0 * s)
    else:
        pop = ssp
    vname = "vsyst_{}".format(name)
    stars = pb.Resample(wave, pb.LOSVDConv(pop, losvdpars=[vname, "sigma"]))
    # Adding a polynomial
    zeroth = True if nssps==1 else False
    poly = pb.Polynomial(wave, porder, zeroth=zeroth, pname=f"p{name}")
    sed = stars * poly
    return sed, limits

def set_priors(parnames, limits, vsyst):
    """ Defining prior distributions for the model. """
    priors = {}
    for parname in parnames:
        name = parname.split("_")[0]
        if name in limits:
            vmin, vmax = limits[name]
            delta = vmax - vmin
            priors[parname] = stats.uniform(loc=vmin, scale=delta)
        elif parname in vsyst:
            priors[parname] = stats.norm(loc=vsyst[parname], scale=500)
        elif parname == "eta":
            priors["eta"] = stats.uniform(loc=1., scale=19)
        elif parname == "nu":
            priors["nu"] = stats.uniform(loc=2, scale=20)
        elif parname == "sigma":
            priors["sigma"] = stats.uniform(loc=50, scale=300)
        elif name in ["pred", "pblue"]:
            porder = int(parname.split("_")[1])
            if porder == 0:
                mu, sd = 1, 1
                a, b = (0 - mu) / sd, (np.infty - mu) / sd
                priors[parname] = stats.truncnorm(a, b, mu, sd)
            else:
                priors[parname] = stats.norm(0, 0.05)
        else:
            print(f"parameter without prior: {parname}")
    return priors

def log_probability(theta):
    """ Calculates the probability of a model."""
    global priors
    global logp
    lp = np.sum([priors[p].logpdf(x) for p, x in zip(logp.parnames, theta)])
    if not np.isfinite(lp) or np.isnan(lp):
        return -np.inf
    ll = logp(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

def run_sampler(outdb, nsteps=5000):
    global logp
    global priors
    ndim = len(logp.parnames)
    nwalkers = 2 * ndim
    pos = np.zeros((nwalkers, ndim))
    logpdf = []
    for i, param in enumerate(logp.parnames):
        logpdf.append(priors[param].logpdf)
        pos[:, i] = priors[param].rvs(nwalkers)
    backend = emcee.backends.HDFBackend(outdb)
    backend.reset(nwalkers, ndim)
    try:
        pool_size = context.mp_pool_size
    except:
        pool_size = 1
    pool = mp.Pool(pool_size)
    with pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                         backend=backend, pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=True)
    return

def run_testdata(dlam=100, nsteps=5000, loglike="studt2", nssps=1,
                 elements=None, target_res=None):
    """ Run paintbox. """
    global logp, priors
    target_res = [200, 100] if target_res is None else target_res
    # List of sky lines to be ignored in the fitting
    skylines = np.array([4792, 4860, 4923, 5071, 5239, 5268, 5577, 5889.99,
                         5895, 5888, 5990, 5895, 6300, 6363, 6386, 6562,
                         6583, 6717, 6730, 7246, 8286, 8344, 8430, 8737,
                         8747, 8757, 8767, 8777, 8787, 8797, 8827, 8836,
                         8919, 9310])
    dsky = 3 # Space around sky lines
    wdir = os.path.join(context.home_dir, "data/testdata/")
    # Providing template file
    pb_dir = os.path.join(wdir, "paintbox-nssps{}-{}".format(nssps, loglike))
    spec = os.path.join(wdir, "NGC7144_spec.fits")
    logps, priors, seds = [], [], []
    waves, fluxes, fluxerrs, masks = [], [], [], []
    wranges = [[4000, 6680], [7800, 8900]]
    for i, side in enumerate(["blue", "red"]):
        tab = Table.read(spec, hdu=i+1)
        #  Normalizing the data to make priors simple
        norm = np.nanmedian(tab["flux"])
        wave = tab["wave"].data
        flux = tab["flux"].data / norm
        fluxerr = tab["fluxerr"].data / norm
        # Setting mask
        # Masks in paintbox are booleans such that True==used in fitting,
        # False==not used
        mask = np.invert(tab["mask"].data.astype(np.bool))
        idx = np.where((wave < wranges[i][0]) | (wave > wranges[i][1]))[0]
        mask[idx] = False
        # Masking all remaining locations where flux is NaN
        mask[np.isnan(flux * fluxerr)] = False
        # Masking lines from Osterbrock atlas
        for line in skylines:
            idx = np.argwhere((wave >= line - dsky) &
                              (wave <= line + dsky)).ravel()
            mask[idx] = False
        # Defining polynomial order
        wmin = wave[mask].min()
        wmax = wave[mask].max()
        porder = int((wmax - wmin) / dlam)
        # Building paintbox model
        outname = f"CvD18_sig{target_res[i]}_{side}"
        sed, limits = make_paintbox_model(wave, outname, nssps=nssps, name=side,
                                  sigma=target_res[i], porder=porder)
        # priors = make_priors(sed.parnames, limits)
        # Defining the likelihood for each part and
        if loglike == "normal":
            logp = pb.NormalLogLike(flux, sed, obserr=fluxerr, mask=mask)
        elif loglike == "studt2":
            logp = pb.StudT2LogLike(flux, sed, obserr=fluxerr, mask=mask)
        logps.append(logp)
        seds.append(sed)
        # logps.append(logp)
        # seds.append(sed)
        # priors.append(priors)
        # waves.append(wave)
        # fluxes.append(flux)
        # fluxerrs.append(fluxerr)
        # masks.append(mask)
    # Make a joint likelihood for all sections
    logp = logps[0]
    for i in range(nssps - 1):
        logp += logps[i+1]
    # Making priors
    v0 = {"vsyst_blue": 1390, "vsyst_red": 1860}
    priors = set_priors(logp.parnames, limits, vsyst=v0)
    # Perform fitting
    dbname = "NGC7144_nssps{}_{}_nsteps{}.h5".format(nssps, loglike, nsteps)
    # Run in any directory outside Dropbox to avoid conflicts
    tmp_db = os.path.join(os.getcwd(), dbname)
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    outdb = os.path.join(pb_dir, dbname)
    if not os.path.exists(outdb):
        run_sampler(tmp_db, nsteps=nsteps)
        if not os.path.exists(pb_dir):
            os.mkdir(pb_dir)
        shutil.move(tmp_db, outdb)

if __name__ == "__main__":
    run_testdata(nssps=2)