""" Run paintbox in observed data. """
import os
import shutil
import copy

import numpy as np
from scipy import stats
import multiprocessing as mp
from astropy.table import Table, hstack, vstack
import emcee
import matplotlib.pyplot as plt
from tqdm import tqdm
import paintbox as pb
from paintbox.utils import CvD18, disp2vel

import context

def make_paintbox_model(wave, name="test", porder=45, nssps=1,
                        sigma=100):
    # Directory where you store your CvD models
    base_dir = context.cvd_dir
    # Locationg where pre-processed models will be stored for paintbox
    outdir = os.path.join(context.home_dir,
                          f"templates/CvD18_sig{sigma}_{name}")
    # Defining wavelength for templates
    velscale = sigma / 2
    wmin = wave.min() - 200
    wmax = wave.max() + 50
    twave = disp2vel([wmin, wmax], velscale)
    ssp = CvD18(twave, sigma=sigma, store=outdir)
    limits = ssp.limits
    if nssps > 1:
        for i in range(nssps):
            p0 = pb.Polynomial(twave, 0, pname="w")
            p0.parnames = [f"w_{i+1}"]
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
    poly = pb.Polynomial(wave, porder, zeroth=True, pname=f"p{name}")
    sed = stars * poly
    return sed, limits

def set_priors(parnames, limits, vsyst, nssps=1):
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
        elif name == "w":
            priors[parname] = stats.uniform(loc=0, scale=1)
        elif name in ["pred", "pblue"]:
            porder = int(parname.split("_")[1])
            if porder == 0:
                mu, sd = 1 / nssps, 1
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

def weighted_traces(parnames, trace, nssps):
    """ Combine SSP traces to have mass/luminosity weighted properties"""
    weights = np.array([trace["w_{}".format(i+1)].data for i in range(
        nssps)])
    wtrace = []
    for param in parnames:
        data = np.array([trace["{}_{}".format(param, i+1)].data
                         for i in range(nssps)])
        t = np.average(data, weights=weights, axis=0)
        wtrace.append(Table([t], names=["{}_weighted".format(param)]))
    return hstack(wtrace)

def make_table(trace, outtab):
    data = np.array([trace[p].data for p in trace.colnames]).T
    v = np.percentile(data, 50, axis=0)
    vmax = np.percentile(data, 84, axis=0)
    vmin = np.percentile(data, 16, axis=0)
    vuerr = vmax - v
    vlerr = v - vmin
    tab = []
    for i, param in enumerate(trace.colnames):
        t = Table()
        t["param"] = [param]
        t["median"] = [round(v[i], 5)]
        t["lerr".format(param)] = [round(vlerr[i], 5)]
        t["uerr".format(param)] = [round(vuerr[i], 5)]
        tab.append(t)
    tab = vstack(tab)
    tab.write(outtab, overwrite=True)
    return tab

def plot_fitting(waves, fluxes, fluxerrs, masks, seds, trace, output,
                 skylines=None, dsky=3):
    width_ratios = [w[-1]-w[0] for w in waves]
    fig, axs = plt.subplots(2, len(seds), gridspec_kw={'height_ratios': [2, 1],
                            "width_ratios": width_ratios},
                            figsize=(2 * context.fig_width, 3))
    for i in range(len(waves)):
        sed = seds[i]
        t = np.array([trace[p].data for p in sed.parnames]).T
        n = len(t)
        wave = waves[i][masks[i]]
        flux = fluxes[i][masks[i]]
        fluxerr = fluxerrs[i][masks[i]]
        models = np.zeros((n, len(wave)))
        y = np.percentile(models, 50, axis=(0,))
        for j in tqdm(range(len(trace)), desc="Generating models "
                                                         "for trace"):
            models[j] = seds[i](t[j])[masks[i]]
        y = np.percentile(models, 50, axis=(0,))
        yuerr = np.percentile(models, 84, axis=(0,)) - y
        ylerr = y - np.percentile(models, 16, axis=(0,))
        ax0 = fig.add_subplot(axs[0,i])
        ax0.errorbar(wave, flux, yerr=fluxerr, fmt="-",
                     ecolor="0.8", c="tab:blue")
        ax0.plot(wave, y, c="tab:orange")
        ax0.xaxis.set_ticklabels([])
        ax0.set_ylabel("Flux")

        ax1 = fig.add_subplot(axs[1,i])
        ax1.errorbar(wave, 100 * (flux - y) / flux, yerr=100 * fluxerr, \
                                                                fmt="-",
                     ecolor="0.8", c="tab:blue")
        ax1.plot(wave, 100 * (flux - y) / flux, c="tab:orange")
        ax1.set_ylabel("Res. (\%)")
        ax1.set_xlabel("$\lambda$ (Angstrom)")
        ax1.set_ylim(-5, 5)
        ax1.axhline(y=0, ls="--", c="k")
        # Include sky lines shades
        if skylines is not None:
            for ax in [ax0, ax1]:
                w0, w1 = ax0.get_xlim()
                for skyline in skylines:
                    if (skyline < w0) or (skyline > w1):
                        continue
                    ax.axvspan(skyline - 3, skyline + 3, color="0.9",
                               zorder=-100)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.show()
    plt.clf()
    plt.close(fig)
    return

def run_paintbox(galaxy, dlam=100, nsteps=5000, loglike="normal2", nssps=1,
                 target_res=None):
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
    wdir = os.path.join(context.home_dir, f"data/{galaxy}")
    # Providing template file
    spec = os.path.join(wdir, f"{galaxy}_spec.fits")
    logps = []
    wranges = [[4000, 6680], [7800, 8900]]
    waves, fluxes, fluxerrs, masks, seds = [], [], [], [], []
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
        sed, limits = make_paintbox_model(wave, nssps=nssps, name=side,
                                  sigma=target_res[i], porder=porder)
        logp = pb.Normal2LogLike(flux, sed, obserr=fluxerr, mask=mask)
        logps.append(logp)
        waves.append(wave)
        fluxes.append(flux)
        fluxerrs.append(fluxerr)
        masks.append(mask)
        seds.append(sed)
    # Make a joint likelihood for all sections
    logp = logps[0]
    for i in range(nssps - 1):
        logp += logps[i+1]
    # Making priors
    v0 = {"vsyst_blue": 1390, "vsyst_red": 1860}
    priors = set_priors(logp.parnames, limits, vsyst=v0, nssps=nssps)
    # Perform fitting
    dbname = f"{galaxy}_nssps{nssps}_{loglike}_nsteps{nsteps}.h5"
    # Run in any directory outside Dropbox to avoid problems
    tmp_db = os.path.join(os.getcwd(), dbname)
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    outdb = os.path.join(wdir, dbname)
    if not os.path.exists(outdb):
        run_sampler(tmp_db, nsteps=nsteps)
        shutil.move(tmp_db, outdb)
    # Post processing of data
    if context.node in context.lai_machines: #not allowing post-processing @LAI
        return
    reader = emcee.backends.HDFBackend(outdb)
    tracedata = reader.get_chain(discard=int(nsteps * 0.9), flat=True, thin=50)
    trace = Table(tracedata, names=logp.parnames)
    if nssps > 1:
        ssp_pars = list(limits.keys())
        wtrace = weighted_traces(ssp_pars, trace, nssps)
        trace = hstack([trace, wtrace])
    outtab = os.path.join(outdb.replace(".h5", "_results.fits"))
    make_table(trace, outtab)
    # # Plot fit
    outimg = outdb.replace(".h5", "_fit.png")
    plot_fitting(waves, fluxes, fluxerrs, masks, seds, trace, outimg,
                 skylines=skylines)


if __name__ == "__main__":
    galaxies = ["NGC7144"]
    for galaxy in galaxies:
        run_paintbox(galaxy, nssps=2)