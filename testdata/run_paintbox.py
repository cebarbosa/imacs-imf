""" Run paintbox in observed data. """
import os
import shutil
import copy

import numpy as np
from scipy import stats
from astropy.io import fits
from astropy.table import Table, vstack, hstack
import matplotlib.pyplot as plt
from tqdm import tqdm
import emcee
import arviz as az
import paintbox as pb

import context

class JointLogLike():
    def __init__(self, ll1, ll2):
        self.ll1 = ll1
        self.ll2 = ll2
        self.parnames = list(dict.fromkeys(ll1.parnames + ll2.parnames))
        self._idxs = []
        for parlist in [ll1.parnames, ll2.parnames]:
            idxs = []
            for p in parlist:
                idxs.append(self.parnames.index(p))
            self._idxs.append(np.array(idxs))

    def __call__(self, theta):
        return self.ll1(theta[self._idxs[0]]) + self.ll2(theta[self._idxs[1]])

def build_sed_CvD(wave, templates_dir, porder=45,
                  elements=None, V=0, polynames="p", vname="vsyst", nssps=1):
    elements = ["C", "N", "Na", "Mg", "Si", "Ca", "Ti", "Fe", "K", "Cr",
                "Mn", "Ba", "Ni", "Co", "Eu", "Sr", "V", "Cu"] if \
                elements is None else elements
    ssp_file = [_ for _ in os.listdir(templates_dir) if _.startswith("VCJ")][0]
    ssp_file = os.path.join(templates_dir, ssp_file)
    templates = fits.getdata(ssp_file, ext=0)
    tnorm = np.median(templates, axis=1)
    templates /= tnorm[:, None]
    params = Table.read(ssp_file, hdu=1)
    twave = Table.read(ssp_file, hdu=2)["wave"].data
    priors = {}
    limits = {}
    for i, param in enumerate(params.colnames):
        vmin, vmax = params[param].min(), params[param].max()
        limits[param] = (vmin, vmax)
        priors[param] = stats.uniform(loc=vmin, scale=vmax-vmin)
    ssp = pb.ParametricModel(twave, params, templates)
    for element in elements:
        elem_file = os.path.join(templates_dir, "C18_rfs_{}.fits".format(
            element))
        if not os.path.exists(elem_file):
            print("Response function for {} not found.".format(element))
            continue
        rfdata = fits.getdata(elem_file, ext=0)
        rfpar = Table.read(elem_file, hdu=1)
        vmin, vmax = rfpar[element].min(), rfpar[element].max()
        limits[element] = (vmin, vmax)
        priors[element] = stats.uniform(loc=vmin, scale=vmax-vmin)
        ewave = Table.read(elem_file, hdu=2)["wave"].data
        rf = pb.ParametricModel(ewave, rfpar, rfdata)
        ssp = ssp * rf
    # For mutltiple stellar populations
    if nssps > 1:
        for i in range(nssps):
            w = pb.Polynomial(twave, 0)
            wname = "w_{}".format(i+1)
            w.parnames = [wname]
            priors[wname] = stats.uniform(loc=0, scale=2)
            s = copy.deepcopy(ssp)
            s.parnames = ["{}_{}".format(_, i+1) for _ in s.parnames]
            for pold, pnew in zip(ssp.parnames, s.parnames):
                priors[pnew] = priors[pold]
            if i == 0:
                pop = w * s
            else:
                pop += (w * s)
    else:
        pop = ssp
    # Adding extinction to the stellar populations
    stars = pb.Resample(wave, pb.LOSVDConv(pop, losvdpars=[vname, "sigma"]))
    priors[vname] = stats.uniform(loc=V - 1000, scale=2000)
    priors["sigma"] = stats.uniform(loc=50, scale=200)
    # Adding a polynomial
    poly = pb.Polynomial(wave, porder)
    poly.parnames = [_.replace("p", polynames) for _ in poly.parnames]
    for p in poly.parnames:
        if p == "{}0".format(polynames):
            mu, sd = 1, 0.3
            a, b = (0 - mu) / sd, (np.infty - mu) / sd
            priors[p] = stats.truncnorm(a, b, mu, sd)
        else:
            priors[p] = stats.norm(0, 0.05)
    # Creating a model including LOSVD
    sed = stars * poly
    sed = pb.Constrain(sed)
    missing = [p for p in sed.parnames if p not in priors.keys()]
    if len(missing) > 0:
        print("Missing parameters in priors: ", missing)
    else:
        print("No missing parameter in the model priors!")
    # Setting properties that may be useful later in modeling
    sed.ssppars = limits
    sed.sspcolnames = params.colnames + elements
    sed.sspparams = params
    sed.porder = porder
    return sed, priors

def run_sampler(loglike, priors, outdb, nsteps=3000):
    ndim = len(loglike.parnames)
    nwalkers = 2 * ndim
    pos = np.zeros((nwalkers, ndim))
    logpdf = []
    for i, param in enumerate(loglike.parnames):
        logpdf.append(priors[param].logpdf)
        pos[:, i] = priors[param].rvs(nwalkers)
    def log_probability(theta):
        lp = np.sum([prior(val) for prior, val in zip(logpdf, theta)])
        if not np.isfinite(lp) or np.isnan(lp):
            return -np.inf
        ll = loglike(theta)
        if not np.isfinite(ll):
            print("nu={}".format(theta[loglike.parnames.index("nu")]), ll)
            return -np.inf
        return lp + ll
    backend = emcee.backends.HDFBackend(outdb)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    backend=backend)
    sampler.run_mcmc(pos, nsteps, progress=True)
    return

def weighted_traces(trace, sed, weights, outtab, redo=False):
    """ Combine SSP traces to have mass/luminosity weighted properties"""
    if os.path.exists(outtab) and not redo:
        a = Table.read(outtab)
        return a
    ws, ps = [], []
    for i in range(sed.nssps):
        w = trace[:, sed.parnames.index("w_{}".format(i+1))]
        j = [sed.parnames.index("{}_{}".format(p, i+1)) for p in
             sed.sspcolnames]
        ssp = trace[:, j]
        n = np.array([weights(s)[0] for s in ssp])
        ws.append(w * n)
        ps.append(ssp)
    ws = np.stack(ws)
    ws /= ws.sum(axis=0)
    ps = np.stack(ps)
    a = np.zeros((len(sed.sspcolnames), len(ws[1])))
    for i, param in enumerate(sed.sspcolnames):
        a[i] = np.sum(ps[:,:,i] * ws, axis=0)
    a = Table(a.T, names=sed.sspcolnames)
    a.write(outtab, overwrite=True)
    return a

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
        ax0 = plt.subplot(axs[0,i])
        ax0.errorbar(wave, flux, yerr=fluxerr, fmt="-",
                     ecolor="0.8", c="tab:blue")
        ax0.plot(wave, y, c="tab:orange")
        ax0.xaxis.set_ticklabels([])
        ax0.set_ylabel("Flux")

        ax1 = plt.subplot(axs[1,i])
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

def weighted_traces(trace):
    """ Combine SSP traces to have mass/luminosity weighted properties"""
    nssps = len([_ for _ in trace.colnames if _.startswith("w_")])
    parnames = ["_".join(_.split("_")[:-1]) for _ in trace.colnames if
                         _.endswith("_1") and _ != "w_1"]
    weights = np.array([trace["w_{}".format(i+1)].data for i in range(nssps)])
    wtrace = []
    for param in parnames:
        data = np.array([trace["{}_{}".format(param, i+1)].data
                         for i in range(nssps)])
        t = np.average(data, weights=weights, axis=0)
        wtrace.append(Table([t], names=["{}_weighted".format(param)]))
    return hstack(wtrace)

def run_testdata(dlam=100, nsteps=5000, loglike="studt2", nssps=1):
    """ Run paintbox. """
    # List of sky lines to be ignored in the fitting
    skylines = np.array([4792, 4860, 4923, 5071, 5239, 5268, 5577, 5889.99,
                         5895, 5888, 5990, 5895, 6300, 6363, 6386, 6562,
                         6583, 6717, 6730, 7246, 8286, 8344, 8430, 8737,
                         8747, 8757, 8767, 8777, 8787, 8797, 8827, 8836,
                         8919, 9310])
    dsky = 3 # Space around sky lines
    V0s = [1390, 1860]
    wdir = os.path.join(context.home_dir, "data/testdata/")
    pb_dir = os.path.join(wdir, "paintbox")
    spec = os.path.join(wdir, "NGC7144_spec.fits")
    logps, priors, seds = [], [], []
    waves, fluxes, fluxerrs, masks = [], [], [], []
    wranges = [[4200, 6680], [8200, 8900]]
    for i, side in enumerate(["blue", "red"]):
        tab = Table.read(spec, hdu=i+1)
        #  Normalizing the data to make priors simple
        norm = np.nanmedian(tab["flux"])
        wave = tab["wave"].data
        flux = tab["flux"].data / norm
        fluxerr = tab["fluxerr"].data / norm
        # Setting mask
        # Masks in paintbox are booleans, True==used in fitting, False==not used
        mask = np.invert(tab["mask"].data.astype(np.bool))
        # Removing wavelengths larger than 6680 AA in blue part
        idx = np.where((wave < wranges[i][0]) | (wave > wranges[i][1]))[0]
        mask[idx] = False
        # Masking all remaining locations where flux is NaN
        mask[np.isnan(flux * fluxerr)] = False
        # Masking lines from Osterbrock atlas
        for line in skylines:
            idx = np.argwhere((wave >= line - dsky) &
                              (wave <= line + dsky)).ravel()
            mask[idx] = False
        # Providing template file
        templates_dir = os.path.join(context.home_dir,
                                      "templates/imacs-{}".format(side))
        # Defining polynomial order
        idx = np.where(np.isfinite(tab["flux"]))[0]
        porder = int((tab["wave"][idx[-1]] - tab["wave"][idx[0]]) / dlam)
        # Producing models and priors for the fitting
        sed, prior = build_sed_CvD(wave, templates_dir, V=V0s[i],
                                      porder=porder,
                                      polynames="p{}".format(side),
                                      vname="vsyst_{}".format(side),
                                      nssps=nssps)
        # Defining the likelihood for each part and
        if loglike == "normal":
            logp = pb.NormalLogLike(flux, sed, obserr=fluxerr, mask=mask)
        elif loglike == "studt2":
            logp = pb.StudT2LogLike(flux, sed, obserr=fluxerr, mask=mask)

        logps.append(logp)
        seds.append(sed)
        priors.append(prior)
        waves.append(wave)
        fluxes.append(flux)
        fluxerrs.append(fluxerr)
        masks.append(mask)
    # Joining the two loglikelihoods
    logp = JointLogLike(logps[0], logps[1])
    priors = {**priors[0], **priors[1]} # Joining the priors
    if loglike in ["studt2", "normal2"]:
        priors["eta"] = stats.uniform(loc=1., scale=19)
    if loglike.startswith("studt"):
        priors["nu"] = stats.uniform(loc=2, scale=20)
    # Running fit
    dbname = "NGC7144_nssps{}_{}_{}.h5".format(nssps, loglike, nsteps)
    # Run in any directory outside Dropbox to avoid conflicts
    tmp_db = os.path.join(os.getcwd(), dbname)
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    outdb = os.path.join(pb_dir, dbname)
    if not os.path.exists(outdb):
        run_sampler(logp, priors, tmp_db, nsteps=nsteps)
        if not os.path.exists(pb_dir):
            os.mkdir(pb_dir)
        shutil.move(tmp_db, outdb)
    # Load database and make a table with summary statistics
    reader = emcee.backends.HDFBackend(outdb)
    tracedata = reader.get_chain(discard=int(nsteps - 300), flat=True, thin=40)
    trace = Table(tracedata, names=logp.parnames)
    if nssps > 1:
        wtrace = weighted_traces(trace)
        trace = hstack([trace, wtrace])
    outtab = os.path.join(outdb.replace(".h5", "_results.fits"))
    make_table(trace, outtab)
    # Plot fit
    outimg = outdb.replace(".h5", "_fit.png")
    plot_fitting(waves, fluxes, fluxerrs, masks, seds, trace, outimg,
                 skylines=skylines)


if __name__ == "__main__":
    run_testdata(nssps=2)