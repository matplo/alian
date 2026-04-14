from functools import singledispatchmethod
from pathlib import Path

import numpy as np
import ROOT as r

from .logs import set_up_logger
from .utils import linbins, logbins, read_yaml

r.TH1.AddDirectory(False)
r.TH1.SetDefaultSumw2()
r.TH2.SetDefaultSumw2()

registry = {
    r.TH1F: [1, '1', '1F', '1f'],
    r.TH2F: [2, '2', '2F', '2f'],
    r.TH3F: [3, '3', '3F', '3f'],
    r.TH1D: ['1D', '1d'],
    r.TH2D: ['2D', '2d'],
    r.TH3D: ['3D', '3d'],
    r.TH1I: ['1I', '1i'],
    r.TH2I: ['2I', '2i'],
    r.TH3I: ['3I', '3i'],
}

# reverse registry mapping to map key to specific ROOT class
HISTOGRAM_REGISTRY = {
    key: value
    for value, keys in registry.items()
    for key in keys
}

class Output:
    def __init__(self, bins, histograms, branches, trees):
        self.logger = set_up_logger(__name__)
        self.logger.info("Initializing histograms...", stacklevel = 2)
        self._init_bins(bins)
        self._init_histograms(histograms)
        self.logger.info("Histograms initialized.", stacklevel = 2)
        self.logger.info("Initializing trees...", stacklevel = 2)
        self._init_branches(branches)
        self._init_trees(trees)
        self.logger.info("Trees initialized.", stacklevel = 2)

    @singledispatchmethod
    @classmethod
    def load(cls, file):
        raise NotImplementedError(f'Cannot configure output from type {type(file)}.')
    @load.register(dict)
    @classmethod
    def _load(cls, cfg):
        if "output" not in cfg:
            raise KeyError("Output must be configured in a 'output' block in the YAML configuration!")
        if cfg["output"] is None:
            raise KeyError("The 'output' block in the YAML configuration cannot be empty!")

        cfg_output = {}
        fields = ["bins", "histograms", "branches", "trees"]
        for field in fields:
            if field not in cfg["output"] or cfg["output"][field] is None:
                cfg_output[field] = {}
            else:
                cfg_output[field] = cfg["output"][field]

        return cls(**cfg_output)
    @load.register(str)
    @load.register(Path)
    @classmethod
    def _load(cls, file):
        cfg = read_yaml(file)
        return cls.load(cfg)

    def _init_bins(self, bins_cfg):
        self._bins = {}
        self._nbins = {}
        for name, arglist in bins_cfg.items():
            self._bins[name] = self._calculate_bins(*arglist)
            self._nbins[name] = len(self._bins[name]) - 1

    def _calculate_bins(self, binning, *bin_info):
        match binning:
            case 'logarithmic' | 'logarithm' | 'log' | 'lg':
                binmin, binmax, nbins = bin_info
                return logbins(binmin, binmax, nbins)
            case 'linear' | 'line' | 'lin' | 'ln':
                binmin, binmax, nbins = bin_info
                return linbins(binmin, binmax, nbins)
            case 'custom' | 'cst' | 'c':
                return np.array(bin_info[0], dtype = float)
            case _:
                raise ValueError(f"Binning type {binning} invalid.")

    def _init_histograms(self, hists_cfg):
        self.hists = {}
        names = []
        # for now, doesn't support nested structures, but could in the future
        for tag, cfg in hists_cfg.items():
            htype, name, title, *bin_names = cfg
            self._validate_bin_names(bin_names)
            # interlace number of bins and bin arrays for each axis
            binnings = [val for name in bin_names for val in (self._nbins[name], self._bins[name])]
            root_hist_cls = HISTOGRAM_REGISTRY[htype]
            self.hists[tag] = root_hist_cls(name, title, *binnings)
            names.append(name)
        duplicates = {name: names.count(name) for name in names if names.count(name) > 1}
        if duplicates:
            self.logger.error(f"Multiple ROOT histograms defined with the same name, they will overwrite each other: {list(duplicates.keys())}")

    def _validate_bin_names(self, bin_names):
        """Check bin names from Histogram configs are defined."""
        invalid_names = [name for name in bin_names if name not in self._bins]
        if invalid_names:
            raise KeyError(f"The following binnings are undefined: {', '.join(invalid_names)}")

    def _init_branches(self, branches):
        # TODO: implement TTrees as output objects
        pass

    def _init_trees(self, trees):
        # TODO: implement TTrees as output objects
        self.trees = {}

    def save(self, tfile):
        """Write output to an *already-opened* TFile."""
        for h in self.hists.values():
            tfile.WriteTObject(h)
        for t in self.trees.values():
            tfile.WriteTObject(t)
