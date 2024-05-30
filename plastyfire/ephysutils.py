"""
Electophysiology utilities (EPSP slope and amplitude calculations)
authors: Giuseppe Chindemi (12.2020) + minor modifications and docs by András Ecker (05.2024)
"""

import h5py
import numpy as np

SPIKE_THRESHOLD = -30  # mV
MIN2MS = 60 * 1000.


def get_epsp_vector(t, v, spikes, window):
    """Extract EPSPs at time `spikes` from voltage trace `v`"""
    # Verify presence of only one EPSP in window
    assert np.max(np.diff(spikes)) > window
    # Get EPSPs
    n = len(spikes)
    epsps = np.zeros(n)
    for i in range(n):
        w0 = np.searchsorted(t, spikes[i])  # Beginning of EPSP window
        w1 = np.searchsorted(t, spikes[i] + window)  # End of EPSP window
        v_baseline = v[w0]
        v_epsp = v[np.argmax(v[w0:w1]) + w0]
        if v_epsp > -SPIKE_THRESHOLD:  # Abort if postsynaptic spike
            raise RuntimeError("Postsynaptic cell spiking during connectivity test")
        epsp = v_epsp - v_baseline
        epsps[i] = epsp
    return epsps


def epsp_slope(vtrace):
    """Calculates the slope of EPSP (rise, from 30% to 75%)"""
    v = vtrace - vtrace[0]  # Remove baseline
    peak = np.max(v)
    peak_idx = np.argmax(v)
    # Get 30% and 75% rise time indices
    idx0 = np.argmin(np.abs(v[:peak_idx] - 0.3 * peak))
    idx1 = np.argmin(np.abs(v[:peak_idx] - 0.75 * peak))
    # Get slope of fitting line between the two
    m = ((v[idx1] - v[idx0]) / (idx1 - idx0))
    return m


class Experiment(object):
    """
    A full STDP induction experiment.
    The experiment consists of two connectivity tests (C01 and C02), separated by the induction protocol.
    """
    def __init__(self, data, c01duration=40., c02duration=40., period=10., c01period=None, c02period=None):
        if type(data) == dict:
            self.t = data["t"]
            self.v = data["v"]
            self.spikes = data["prespikes"]
        elif type(data) == str:
            h5file = h5py.File(data, "r")
            self.t = h5file["t"][()]
            self.v = h5file["v"][()]
            self.spikes = h5file["prespikes"][()]
            h5file.close()
        else:
            raise Exception
        # Store other attributes
        self.duration = {"C01": c01duration * MIN2MS,
                         "C02": c02duration * MIN2MS}
        self.period = period * 1000.  # sec to ms
        self.c01period = c01period * 1000. if c01period is not None else period * 1000.
        self.c02period = c02period * 1000. if c02period is not None else period * 1000.
        # Create common attributes
        self.epspwindow = 100.  # ms
        self.cxs = ["C01", "C02"]
        self.cxspikes = {"C01": self.spikes[:int(self.duration["C01"] / self.c01period)],
                         "C02": self.spikes[-int(self.duration["C02"] / self.c02period):]}
        # Initialize lazy-loading properties
        self._epsp = None
        self._cxtrace = None

    @property
    def epsp(self):
        if self._epsp is None:
            self._epsp = {cx: get_epsp_vector(self.t, self.v, self.cxspikes[cx], self.epspwindow) for cx in self.cxs}
        return self._epsp

    @property
    def cxtrace(self, dt_int=0.025):
        if self._cxtrace is None:
            tdense = np.linspace(0, self.epspwindow, int(self.epspwindow / dt_int))
            self._cxtrace = {"t": tdense}
            for cx in self.cxs:
                self._cxtrace[cx] = []
                for s in self.cxspikes[cx]:
                    idx0 = np.searchsorted(self.t, s)
                    idx1 = np.searchsorted(self.t, s + self.epspwindow)
                    vdense = np.interp(tdense + s, self.t[idx0:idx1], self.v[idx0:idx1])
                    self._cxtrace[cx].append(vdense)
                self._cxtrace[cx] = np.array(self.cxtrace[cx])
        return self._cxtrace

    def compute_epsp_interval(self, interval):
        """Compute mean EPSP amplitude at regular `interval`s (in minutes)."""
        results = {}
        for cx in self.cxs:
            n = int(self.duration[cx] / (interval * MIN2MS))
            epsp_groups = np.split(self.epsp[cx], n)
            spike_groups = np.split(self.cxspikes[cx], n)
            avg = np.mean(epsp_groups, axis=1)
            sem = np.std(epsp_groups, axis=1) / np.sqrt(len(epsp_groups[0]))
            t = np.mean(spike_groups, axis=1)
            results[cx] = {"avg": avg, "sem": sem, "t": t}
        return results

    def compute_epsp_ratio(self, n, method="amplitude", full=False):
        """Compute mean EPSP change."""
        if method == "amplitude":
            epsp_before = np.mean(self.epsp["C01"][-n:])
            epsp_before_std = np.std(self.epsp["C01"][-n:])
            epsp_after = np.mean(self.epsp["C02"][-n:])
            epsp_after_std = np.std(self.epsp["C02"][-n:])
        elif method == "slope":
            epsp_before = epsp_slope(np.mean(self.cxtrace["C01"][-n:], axis=0))
            epsp_before_std = 0
            epsp_after = epsp_slope(np.mean(self.cxtrace["C02"][-n:], axis=0))
            epsp_after_std = 0
        else:
            raise ValueError("Unknown method %s" % method)
        epsp_ratio = epsp_after / epsp_before
        if full:
            return epsp_before, epsp_after, epsp_ratio, epsp_before_std, epsp_after_std
        else:
            return epsp_ratio

    def normalize_time(self, t):
        """
        Normalize time vector `t`.
        (Convert milliseconds to minutes and shift t = 0 to beginning of induction phase)
        """
        return (t - self.duration["C01"]) / MIN2MS

