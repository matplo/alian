from functools import singledispatchmethod
from pathlib import Path

import heppyy

from .logs import set_up_logger
from .utils import read_yaml

alian = heppyy.load_cppyy("alian")
fj = heppyy.load_cppyy('fastjet')

class JetFinder:
    """Helper class for jet finding.

    This class stores jet finding parameters, initialized via YAML, and returns jets with
    `find_jets()`

    Attributes:
        R (str): The jet resolution / radius parameter.
        alg (fastjet.JetDefinition): The jet definition.
        pT_min (int): The minimum jet transverse momentum in GeV.
    """

    _defaults = {
        'R': 0.4,
        'alg': fj.antikt_algorithm,
        'pT_min': 10,
    }
    _defaults["eta_max"] = 0.9 - _defaults["R"]

    alg_registry = {
        fj.antikt_algorithm: ['antikt', 'anti-kt', 'antikT', 'anti-kT', 'anti', 'ak', -1],
        fj.kt_algorithm: ['kt', 'kT', 'k', 1],
        fj.cambridge_algorithm: ['ca', 'CA', 'cambridge' 'cambridge-aachen', 'cambridge/aachen', 0],
    }
    alg_map = { # mapping from string to fj.JetAlgorithm
        key: value
        for value, keys in alg_registry.items()
        for key in keys
    }

    def __init__(self, alg = fj.antikt_algorithm, R = 0.4, eta_max = None, pT_min = 10):
        self.logger = set_up_logger(__name__)
        self.R = R
        if eta_max is None:
            self.eta_max = 0.9 - self.R
        else:
            self.eta_max = eta_max
        self.alg = self._parse_jet_alg(alg)
        self.jet_def = fj.JetDefinition(self.alg, self.R)
        self.pT_min = pT_min
        self.jet_selector = fj.SelectorAbsEtaMax(self.eta_max) * fj.SelectorPtMin(pT_min)
        fj.ClusterSequence.print_banner()
        # TODO: jet background subtraction
        # self.area_def = fj.AreaDefinition(fj.active_area, fj.GhostedAreaSpec(self.eta_max + self.jet_R))
        # self.bg_estimator = fj.GridMedianBackgroundEstimator(self.bg_y_max, self.bg_grid_spacing)

    @singledispatchmethod
    @classmethod
    def load(cls, file, *args, **kwargs):
        raise NotImplementedError(f'Cannot configure jet finder from type {type(file)}.')
    @load.register(dict)
    @classmethod
    def _load(cls, cfg):
        if "jet_finder" not in cfg:
            raise KeyError("The JetFinder must be configured in a 'jet_finder' block in the YAML configuration!")
        if cfg['jet_finder'] is None:
            cfg["jet_finder"] = {}

        options = {**cls._defaults, **cfg["jet_finder"]}

        return cls(**options)
    @load.register(str)
    @load.register(Path)
    @classmethod
    def _load(cls, file):
        cfg = read_yaml(file)
        return cls.load(cfg)

    def dump(self):
        """Dump all jet finding parameters."""
        cfg = "\n".join([f"\t{param}: {repr(getattr(self, param))}" for param in self._defaults])
        self.logger.info(f"JetFinder configuration:\n{cfg}", stacklevel = 2)

    def _parse_jet_alg(self, alg: str):
        if isinstance(alg, fj.JetAlgorithm):
            return alg
        elif isinstance(alg, str):
            try:
                return self.alg_map[alg]
            except KeyError:
                valid_algs = sum(self.alg_registry.values(), [])
                self.logger.error(f"Did not recognize jet algorithm '{alg}', defaulting to anti-kT. Valid options: {valid_algs}")
                return fj.antikt_algorithm
        else:
            valid_algs = sum(self.alg_registry.values(), [])
            self.logger.error(f"Cannot pass type {type(alg)} as jet algorithm, defaulting to anti-kT. Valid options: {valid_algs}")
            return fj.antikt_algorithm

    def find_jets(self, tracks, m = 0.13957, index_offset = 0):
        # NOTE: ClusterSequence must be attached somehow to self to keep it in scope
        self.cs = fj.ClusterSequence(tracks, self.jet_def)
        jets = fj.sorted_by_pt(self.jet_selector(self.cs.inclusive_jets()))
        return jets
