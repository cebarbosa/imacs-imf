""" Run paintbox in observed data. """
import os
import shutil

import numpy as np
from scipy import stats
from astropy.io import fits
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
import emcee
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

def build_sed_CvD(wave, templates_dir, velscale=200, porder=45,
                  elements=None, V=0, polynames="p"):
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
    # Adding extinction to the stellar populations
    stars = pb.Resample(wave, pb.LOSVDConv(ssp, velscale=velscale,
                                           losvdpars=["vsyst", "sigma"]))
    priors["vsyst"] = stats.norm(loc=V, scale=100)
    priors["sigma"] = stats.uniform(loc=50, scale=450)
    # Adding a polynomial
    poly = pb.Polynomial(wave, porder)
    poly.parnames = [_.replace("p", polynames) for _ in poly.parnames]
    for p in poly.parnames:
        if p == "{}0".format(polynames):
            mu, sd = 1, 0.3
            a, b = (0 - mu) / sd, (np.infty - mu) / sd
            priors[p] = stats.truncnorm(a, b, mu, sd)
        else:
            priors[p] = stats.norm(0, 0.02)
    # Creating a model including LOSVD
    sed = stars * poly
    sed = pb.ConstrainModel(sed)
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
        return lp + loglike(theta)
    backend = emcee.backends.HDFBackend(outdb)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                    backend=backend)
    sampler.run_mcmc(pos, nsteps, progress=True)
    return

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

def plot_fitting(waves, fluxes, fluxerrs, masks, seds, colnames, trace):
    for i in range(len(waves)):
        idx = [colnames.index(p) for p in seds[i].parnames]
        t = trace[:, idx]
        p = t.mean(axis=0)
        plt.errorbar(waves[i][masks[i]], fluxes[i][masks[i]],
                     yerr=fluxerrs[i][masks[i]], fmt="-",
                     ecolor="0.8", c="tab:blue")
        plt.plot(waves[i][masks[i]], seds[i](p)[masks[i]], c="tab:orange")
    plt.show()

def run_testdata():
    V0 = 1930
    dlam = 150 # Used to define polynomial order
    target_res = np.array([200, 100]) # Rounding up the ideal resolution
    velscale = np.ceil(target_res / 2.5)
    wdir = os.path.join(context.home_dir, "data/testdata/")
    spec = os.path.join(wdir, "NGC7144_spec.fits")
    logps, priors, seds = [], [], []
    waves, fluxes, fluxerrs, masks = [], [], [], []
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
        if i == 0:
            idx = np.where(wave > 6680)[0]
            mask[idx] = False
        # Masking all remaining locations where flux is NaN
        mask[np.isnan(flux * fluxerr)] = False
        # Providing template file
        templates_dir = os.path.join(context.home_dir,
                                      "templates/imacs-{}".format(side))
        # Defining polynomial order
        idx = np.where(np.isfinite(tab["flux"]))[0]
        porder = int((tab["wave"][idx[-1]] - tab["wave"][idx[0]]) / dlam)
        # Producing models and priors for the fitting
        sed, prior = build_sed_CvD(wave, templates_dir, V=V0,
                                      velscale=velscale[i], porder=porder,
                                      polynames="p{}".format(side))

        # Defining the likelihood for each part and
        logp = pb.NormalLogLike(flux, sed, obserr=fluxerr, mask=mask)
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
    # Running fit
    # Run in any directory outside Dropbox to avoid conflicts
    tmp_db = os.path.join(os.getcwd(), os.path.split(spec)[1].replace(".fits",
                                                                 ".h5"))
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    outdb = spec.replace(".fits", ".h5")
    nsteps = 3000
    if not os.path.exists(outdb):
        run_sampler(logp, priors, tmp_db, nsteps=nsteps)
        shutil.move(tmp_db, outdb)
    # Load database and make a table with summary statistics
    reader = emcee.backends.HDFBackend(outdb)
    tracedata = reader.get_chain(discard=int(.9 * nsteps), flat=True, thin=40)
    trace = Table(tracedata, names=logp.parnames)
    outtab = os.path.join(outdb.replace(".h5", "_results.fits"))
    make_table(trace, outtab)
    # Plot fit
    plot_fitting(waves, fluxes, fluxerrs, masks, seds, logp.parnames, tracedata)


if __name__ == "__main__":
    run_testdata()
