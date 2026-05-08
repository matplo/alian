from functools import singledispatchmethod
from pathlib import Path

import numpy as np

import heppyy

from .logs import set_up_logger
from .selection import EventSel, RCTSel, TrackSel, TrigSel
from .utils import read_yaml

alian = heppyy.load_cppyy("alian")

class Selector:
    _types = {}
    _name = ""
    _dummy_mask = ""
    def __init__(self, **selections):
        self.logger = set_up_logger(type(self).__name__)
        self._masks = {sel: None for sel in self._types.keys()}
        self.reset()
        self.update(**selections)

    @singledispatchmethod
    @classmethod
    def load(cls, file):
        raise NotImplementedError(f'Cannot configure selector from type {type(file)}.')

    @load.register(dict)
    @classmethod
    def _load(cls, cfg):
        if "selections" not in cfg:
            raise KeyError("Selectors must be defined in a 'selections' block in the YAML configuration!")
        if cfg['selections'] is None:
            raise KeyError("The 'selections' block in the YAML configuration cannot be empty!")

        if cls._name in cfg["selections"]:
            if cfg["selections"][cls._name] is None:
                return cls()
            else:
                return cls(**cfg["selections"][cls._name])
        else:
            return None

    @load.register(str)
    @load.register(Path)
    @classmethod
    def _load(cls, file):
        cfg = read_yaml(file)
        return cls.load(cfg)

    def reset(self):
        """Reset all selections."""
        for selection in self._types.keys():
            setattr(self, selection, None)

    def update(self, **selections):
        """Set given selections, leaving all other unspecified selections as-is."""
        valid_sels = self._validate(**selections)
        for name, val in valid_sels.items():
            setattr(self, name, val)

    def _is_ignore_selection(self, value):
        """Check if the selection (written as `value` in the YAML file) should be ignored."""
        return value in [None, "None", "none"]

    @property
    def rdf_mask(self):
        mask = " && ".join([mask for mask in self._masks.values() if mask is not None])
        if not mask: # if no selections active, pass dummy mask
            return self._dummy_mask
        return mask
    def apply_to(self, df):
        """Apply cuts to the event RDataFrame `df`."""
        raise NotImplementedError

    def _validate(self, **selections):
        """Validate the passed selections, and raise warnings if invalid selections were given."""
        # validate selection names
        unrecognized_selections = [sel for sel in selections.keys() if sel not in self._types.keys()]
        if unrecognized_selections:
            raise KeyError(f"Unrecognized selections were given: {', '.join(unrecognized_selections)}")
        # validate selection types
        recognized_selections = {sel: val for sel, val in selections.items() if sel in self._types.keys()}
        # check that each selection value is either the right type, or if is not, then it is an allowed ignore keyword
        wrong_types = [f"{sel}: {val} (type {type(val)}), must be {self._types[sel]}"
                            for sel, val in recognized_selections.items()
                            if not isinstance(val, self._types[sel]) and not self._is_ignore_selection(val) ]
        if wrong_types:
            wrong_types = "\n".join(wrong_types)
            raise TypeError(f"Incorrect types for the following selections were given. Use either the given types or 'none':\n{wrong_types}")
        return recognized_selections

    def _canonicalize(self, sel, val):
        if val in [None, "None", "none"]:
            return None
        if isinstance(val, self._types[sel]):
            return val
        raise TypeError(f"Incorrect type for {type(self).__name__}: type for {sel} must be {self._types[sel]}, not {type(val)}")

    def _pass_ignore(self, *args, **kwargs):
        return True
    def _fail_ignore(self, *args, **kwargs):
        return False

    def selects(self, obj, verbose = False):
        raise NotImplementedError

    def dump(self):
        """Dump all selections."""
        cfg = "\n".join([f"\t{sel}: {getattr(self, sel)}" for sel in self._types.keys()])
        self.logger.info(f"{type(self).__name__} configuration:\n{cfg}", stacklevel = 2)

class EventSelector(Selector):
    """An event selector, configured via YAML.

    The EventSelector contains all event-level selections. To ignore a particular selection,
    set the relevant selection to None. In particular, to analyze MB (i.e. unskimmed data),
    `trig_sel` must be set to `None` to avoid losing all events. Multiple selections for the
    `event_sel`, `trig_sel`, and `rct` can be combined with commas in the config. Be careful
    with "dummy" centrality, multiplicity, and occupancy values! For example, multiplicities
    are sometimes filled with dummy values of -999, so a "safe" minimum multiplicity cut of 0
    might accidentally remove events you don't want to remove.

    Attributes:
        event_sel (EventSel): event selection to be applied. Events must pass ALL given selections.
        trig_sel (TrigSel): trigger selection to be applied. Events must satisfy AT LEAST ONE selection.
        rct (RCTSel): RCT selection to be applied. Events must not be marked Bad for ANY of the relevant detectors.
        cent_min (int/float): minimum centrality.
        cent_max (int/float): maximum centrality.
        mult_min (int/float): minimum multiplicity.
        mult_max (int/float): maximum multiplicity.
        occupancy_min (int/float): minimum occupancy.
        occupancy_max (int/float): maximum occupancy.
    """

    _types = {
        "event_sel": (EventSel, str, int),
        "trig_sel": (TrigSel, str, int),
        "rct": (RCTSel, str, int),
        "cent_min": (int, float),
        "cent_max": (int, float),
        "mult_min": (int, float),
        "mult_max": (int, float),
        "occupancy_min": (int, float),
        "occupancy_max": (int, float),
    }
    _name = "event"
    _dummy_mask = "event_sel >= 0"

    # special handling of no-selects to allow for ignoring the event or trigger selection altogether
    @property
    def event_sel(self):
        return self._event_sel
    @event_sel.setter
    def event_sel(self, event_sel):
        self._event_sel = EventSel.create(self._canonicalize('event_sel', event_sel))
        if self._event_sel is None:
            self.pass_event_sel = self._pass_ignore
            self._masks['event_sel'] = None
        else:
            self.pass_event_sel = self._pass_event_sel
            self._masks['event_sel'] = f"(({int(self._event_sel)} & event_sel) == {int(self._event_sel)})"
    @property
    def trig_sel(self):
        return self._trig_sel
    @trig_sel.setter
    def trig_sel(self, trig_sel):
        self._trig_sel = TrigSel.create(self._canonicalize('trig_sel', trig_sel))
        if self._trig_sel is None:
            self.pass_trig_sel = self._pass_ignore
            self._masks['trig_sel'] = None
        else:
            self.pass_trig_sel = self._pass_trig_sel
            self._masks['trig_sel'] = f"({int(self._trig_sel)} & trig_sel)"
    @property
    def rct(self):
        return self._rct
    @rct.setter
    def rct(self, rct):
        # NOTE: CCDB check and ZDC have to be included in RCT manually (combine with comma in config)
        self._rct = RCTSel.create(self._canonicalize('rct', rct))
        if self._rct is None:
            self.pass_rct = self._pass_ignore
            self._masks['rct'] = None
        else:
            self.pass_rct = self._pass_rct
            self._masks['rct'] = f"!({int(self._rct)} & rct)"
    @property
    def cent_min(self):
        return self._cent_min
    @cent_min.setter
    def cent_min(self, cent_min):
        self._cent_min = self._canonicalize('cent_min', cent_min)
        if self._cent_min is None:
            self.pass_cent_min = self._pass_ignore
            self._masks['cent_min'] = None
        else:
            self.pass_cent_min = self._pass_cent_min
            self._masks['cent_min'] = f"(centrality >= {self._cent_min})"
    @property
    def cent_max(self):
        return self._cent_max
    @cent_max.setter
    def cent_max(self, cent_max):
        self._cent_max = self._canonicalize('cent_max', cent_max)
        if self._cent_max is None:
            self.pass_cent_max = self._pass_ignore
            self._masks['cent_max'] = None
        else:
            self.pass_cent_max = self._pass_cent_max
            self._masks['cent_max'] = f"(centrality <= {self._cent_max})"
    @property
    def mult_min(self):
        return self._mult_min
    @mult_min.setter
    def mult_min(self, mult_min):
        self._mult_min = self._canonicalize('mult_min', mult_min)
        if self._mult_min is None:
            self.pass_mult_min = self._pass_ignore
            self._masks['mult_min'] = None
        else:
            self.pass_mult_min = self._pass_mult_min
            self._masks['mult_min'] = f"(multiplicity >= {self._mult_min})"
    @property
    def mult_max(self):
        return self._mult_max
    @mult_max.setter
    def mult_max(self, mult_max):
        self._mult_max = self._canonicalize('mult_max', mult_max)
        if self._mult_max is None:
            self.pass_mult_max = self._pass_ignore
            self._masks['mult_max'] = None
        else:
            self.pass_mult_max = self._pass_mult_max
            self._masks['mult_max'] = f"(multiplicity <= {self._mult_max})"
    @property
    def occupancy_min(self):
        return self._occupancy_min
    @occupancy_min.setter
    def occupancy_min(self, occupancy_min):
        self._occupancy_min = self._canonicalize('occupancy_min', occupancy_min)
        if self._occupancy_min is None:
            self.pass_occupancy_min = self._pass_ignore
            self._masks['occupancy_min'] = None
        else:
            self.pass_occupancy_min = self._pass_occupancy_min
            self._masks['occupancy_min'] = f"(occupancy >= {self._occupancy_min})"
    @property
    def occupancy_max(self):
        return self._occupancy_max
    @occupancy_max.setter
    def occupancy_max(self, occupancy_max):
        self._occupancy_max = self._canonicalize('occupancy_max', occupancy_max)
        if self._occupancy_max is None:
            self.pass_occupancy_max = self._pass_ignore
            self._masks['occupancy_max'] = None
        else:
            self.pass_occupancy_max = self._pass_occupancy_max
            self._masks['occupancy_max'] = f"(occupancy <= {self._occupancy_max})"

    def apply_to(self, df, label = "selected"):
        """Apply event selections to the event RDataFrame."""
        self.logger.info(f"Applying event mask: {(mask := self.rdf_mask)}")
        return df.Filter(mask)

    def _pass_event_sel(self, event):
        """Check that the event satisfies ALL requested event selection bits."""
        return self._event_sel in event.event_sel
    def _pass_trig_sel(self, event):
        """Check that the event satisfies ANY requested trigger bit."""
        return bool(self._trig_sel & event.trig_sel)
    def _pass_rct(self, event):
        """Check that the event is not Bad for ANY relevant detectors."""
        return not bool(self._rct & event.rct)
    def _pass_cent_min(self, event):
        return event.centrality >= self._cent_min
    def _pass_cent_max(self, event):
        return event.centrality <= self._cent_max
    def _pass_mult_min(self, event):
        return event.multiplicity >= self._mult_min
    def _pass_mult_max(self, event):
        return event.multiplicity <= self._mult_max
    def _pass_occupancy_min(self, event):
        return event.occupancy >= self._occupancy_min
    def _pass_occupancy_max(self, event):
        return event.occupancy <= self._occupancy_max
    def selects(self, event):
        """Determine if event passes all event-level selections: event selection, trigger selection, RCT, centrality, multiplicity, occupancy."""
        return self.pass_event_sel(event) and self.pass_trig_sel(event) and self.pass_rct(event) \
           and self.pass_cent_min(event) and self.pass_cent_max(event) \
           and self.pass_mult_min(event) and self.pass_mult_max(event) \
           and self.pass_occupancy_min(event) and self.pass_occupancy_max(event)

class TrackSelector(Selector):
    """An track selector, configured via YAML.

    The TrackSelector contains all track-level selections. To ignore a particular selection,
    set the relevant selection to None.  Multiple `track_sel` can be combined with
    commas in the config.

    Attributes:
        pt_min (int/float): minimum track transverse momentum.
        track_sel (TrackSel): track selection to be applied. Tracks must satisfy AT LEAST ONE selection.
    """

    _types = {
        "pt_min": (float, int),
        "track_sel": (TrackSel, str, int)
    }
    _name = "track"
    _dummy_mask = "track_pt > -1"

    @property
    def pt_min(self):
        return self._pt_min
    @pt_min.setter
    def pt_min(self, pt_min):
        self._pt_min = self._canonicalize('pt_min', pt_min)
        if self._pt_min is None:
            self.pass_pt_min = self._pass_ignore
            self._masks['pt_min'] = None
        else:
            self.pass_pt_min = self._pass_pt_min
            self._masks['pt_min'] = f"(track_pt >= {self._pt_min})"
    @property
    def track_sel(self):
        return self._track_sel
    @track_sel.setter
    def track_sel(self, track_sel):
        self._track_sel = TrackSel.create(self._canonicalize('track_sel', track_sel))
        if self._track_sel is None:
            self.pass_track_sel = self._pass_ignore
            self._masks['track_sel'] = None
        else:
            self.pass_track_sel = self._pass_track_sel
            self._masks['track_sel'] = f"(track_sel & {int(self._track_sel)})"

    def _pass_pt_min(self, track):
        """Check that the track has the minimum required pT"""
        return track.pt() >= self._pt_min
    def _pass_track_sel(self, track):
        """Check that the track satisfies ANY requested track selection bit."""
        return bool(self._track_sel & track.user_info[alian.TrackInfo]().track_sel())

    def apply_to(self, df, label = "selected"):
        """Apply selections to RDataFrame df."""
        self.logger.info(f"Applying track mask: {(mask := self.rdf_mask)}")
        ret = df.Define("track_mask", mask)
        # create new columns based on the mask
        track_cols = [col for col in df.GetColumnNames() if col.startswith("track_")]
        name_map = {col: f"{col}_{label}" for col in track_cols}
        for colname, masked_colname in name_map.items():
            ret = ret.Define(masked_colname, f"{colname}[track_mask]")
        return ret

    def selects(self, track):
        """Determine if track passes the track selection."""
        return self.pass_track_sel(track) and self.pass_pt_min(track)

    def __call__(self, track):
        return self.selects(track)

class ClusterSelector(Selector):
    """An cluster selector, configured via YAML.

    The ClusterSelector contains all cluster-level selections. To ignore a particular selection,
    set the relevant selection to None. Note that, due to the BerkeleyTree structure, cuts related
    to matched tracks cannot be applied to RDataFrames.

    Attributes:
        energy_min (int/float): minimum cluster energy.
        pt_min (int/float): minimum cluster transverse momentum.
        m02_min (int/float): minimum cluster M02.
        m02_max (int/float): maximum cluster M02.
        ncells_min (int/float): minimum number of cells in the cluster. Cluster must have ncells greater than OR EQUAL TO this value.
        time_min (int/float): minimum cluster time.
        time_max (int/float): maximum cluster time.
        exoticity (bool): exoticity selection. To select non-exotic clusters, set to True. To select exotic clusters, set to False. To ignore exoticity entirely, set to None.
        dbc (int/float): minimum distance to the nearest bad EMC channel. Note that the JE derived data currently fills this with a dummy value of 1024.
        nlm (int): minimum number of local maxima. Cluster must have an NLM greater than OR EQUAL TO this value.
        defn (int): filter on a cluster definition. Set to 0 for V1, 10 for V3. Multiple definitions cannot be filtered on simultaneously; if needed, set to None and check manually on analysis level.
        edge (bool): apply the SM edge cut.
        delta_eta (int/float): maximum difference in eta (at EMC) between cluster and track to match.
        delta_phi (int/float): maximum difference in phi (at EMC) between cluster and track to match.
        ep_max (int/float): maximum ratio of cluster energy to track momentum to match.
    """

    w = 0.0143 # cell granularity in eta/phi
    sm_full = (81.2 * np.pi / 180, np.pi)
    sm_23 = (261.2 * np.pi / 180, 318.8 * np.pi / 180)
    sm_13e = (181.2 * np.pi / 180, 185.8 * np.pi / 180)
    sm_13d = (321.2 * np.pi / 180, 325.8 * np.pi / 180)

    _types = {
        # cluster cuts
        "energy_min": (float, int),
        "pt_min": (float, int),
        "m02_min": (float, int),
        "m02_max": (float, int),
        "ncells_min": (float, int),
        "time_min": (float, int),
        "time_max": (float, int),
        "exoticity": bool,
        "dbc": (float, int),
        "nlm": int,
        "defn": int,
        "edge": bool,
        "delta_eta": (float, int),
        "delta_phi": (float, int),
        "ep_max": (float, int),
    }
    _name = "cluster"
    _dummy_mask = "cluster_pt > -1"

    @property
    def energy_min(self):
        return self._energy_min
    @energy_min.setter
    def energy_min(self, energy_min):
        self._energy_min = self._canonicalize('energy_min', energy_min)
        if self._energy_min is None:
            self.pass_energy_min = self._pass_ignore
            self._masks['energy_min'] = None
        else:
            self.pass_energy_min = self._pass_energy_min
            self._masks['energy_min'] = f"(cluster_energy >= {self._energy_min})"
    @property
    def pt_min(self):
        return self._pt_min
    @pt_min.setter
    def pt_min(self, pt_min):
        self._pt_min = self._canonicalize('pt_min', pt_min)
        if self._pt_min is None:
            self.pass_pt_min = self._pass_ignore
            self._masks['pt_min'] = None
        else:
            self.pass_pt_min = self._pass_pt_min
            self._masks['pt_min'] = f"(cluster_pt >= {self._pt_min})"
    @property
    def m02_min(self):
        return self._m02_min
    @m02_min.setter
    def m02_min(self, m02_min):
        self._m02_min = self._canonicalize('m02_min', m02_min)
        if self._m02_min is None:
            self.pass_m02_min = self._pass_ignore
            self._masks['m02'] = None
        else:
            self.pass_m02_min = self._pass_m02_min
            self._masks['m02'] = f"(cluster_m02 >= {self._m02_min})"
    @property
    def m02_max(self):
        return self._m02_max
    @m02_max.setter
    def m02_max(self, m02_max):
        self._m02_max = self._canonicalize('m02_max', m02_max)
        if self._m02_max is None:
            self.pass_m02_max = self._pass_ignore
            self._masks['m02'] = None
        else:
            self.pass_m02_max = self._pass_m02_max
            self._masks['m02'] = f"(cluster_m02 <= {self._m02_max})"
    @property
    def ncells_min(self):
        return self._ncells_min
    @ncells_min.setter
    def ncells_min(self, ncells_min):
        self._ncells_min = self._canonicalize('ncells_min', ncells_min)
        if self._ncells_min is None:
            self.pass_ncells_min = self._pass_ignore
            self._masks['ncells'] = None
        else:
            self.pass_ncells_min = self._pass_ncells_min
            self._masks['ncells'] = f"(cluster_ncells >= {self._ncells_min})"
    @property
    def time_min(self):
        return self._time_min
    @time_min.setter
    def time_min(self, time_min):
        self._time_min = self._canonicalize('time_min', time_min)
        if self._time_min is None:
            self.pass_time_min = self._pass_ignore
            self._masks['time_min'] = None
        else:
            self.pass_time_min = self._pass_time_min
            self._masks['time_min'] = f"(cluster_time >= {self._time_min})"
    @property
    def time_max(self):
        return self._time_max
    @time_max.setter
    def time_max(self, time_max):
        self._time_max = self._canonicalize('time_max', time_max)
        if self._time_max is None:
            self.pass_time_max = self._pass_ignore
            self._masks['time_max'] = None
        else:
            self.pass_time_max = self._pass_time_max
            self._masks['time_max'] = f"(cluster_time <= {self._time_max})"
    @property
    def exoticity(self):
        return self._exoticity
    @exoticity.setter
    def exoticity(self, exoticity):
        self._exoticity = self._canonicalize('exoticity', exoticity)
        if self._exoticity is None:
            self.pass_exotic = self._pass_ignore
            self._masks['exoticity'] = None
        elif self._exoticity is True:
            self.pass_exotic = self._pass_exotic
            self._masks['exoticity'] = "(!cluster_exoticity)"
        elif self._exoticity is False:
            self.logger.warning("The exoticity cut has been set to False, which will yield exotic clusters only. If you wanted to remove the exoticity check entirely, set this to 'none'.")
            self.pass_exotic = self._fail_exotic
            self._masks['exoticity'] = "(cluster_exoticity)"
        else:
            raise RuntimeError("Somehow the type checking was bypassed here...")
    @property
    def dbc(self):
        return self._dbc
    @dbc.setter
    def dbc(self, dbc):
        self._dbc = self._canonicalize('dbc', dbc)
        if self._dbc is None:
            self.pass_dbc = self._pass_ignore
            self._masks['dbc'] = None
        else:
            self.pass_dbc = self._pass_dbc
            self._masks['dbc'] = f"(cluster_dbc >= {self._dbc})"
    @property
    def nlm(self):
        return self._nlm
    @nlm.setter
    def nlm(self, nlm):
        self._nlm = self._canonicalize('nlm', nlm)
        if self._nlm is None:
            self.pass_nlm = self._pass_ignore
            self._masks['nlm'] = None
        else:
            self.pass_nlm = self._pass_nlm
            self._masks['nlm'] = f"(cluster_nlm <= {self._nlm})"
    @property
    def defn(self):
        return self._defn
    @defn.setter
    def defn(self, defn):
        self._defn = self._canonicalize('defn', defn)
        if self._defn is None:
            self.pass_defn = self._pass_ignore
            self._masks['defn'] = None
        else:
            self.pass_defn = self._pass_defn
            self._masks['defn'] = f"(cluster_defn == {self._defn})"
    @property
    def edge(self):
        return self._edge
    @edge.setter
    def edge(self, edge):
        self._edge = self._canonicalize('edge', edge)
        if not self._edge: # either False or None
            self.pass_edge = self._pass_ignore
            self._masks['edge'] = None
        else:
            self.pass_edge = self._pass_edge
            self._masks['edge'] = ("( (cluster_eta_abs < 0.67 && ("
                    f"(cluster_phi > {self.sm_full[0]} && cluster_phi < {self.sm_full[1]}) || "
                    f"(cluster_phi > {self.sm_13e[0]} && cluster_phi < {self.sm_13e[1]}) || "
                    f"(cluster_phi > {self.sm_13d[0]} && cluster_phi < {self.sm_13d[1]}) )) "
                    f"|| (cluster_eta_abs > 0.25 && cluster_eta_abs < 0.67 && cluster_phi > {self.sm_23[0]} && cluster_phi < {self.sm_23[1]}) )"
            )
    @property
    def delta_eta(self):
        return self._delta_eta
    @delta_eta.setter
    def delta_eta(self, delta_eta):
        self._delta_eta = self._canonicalize('delta_eta', delta_eta)
        if self._delta_eta is None:
            # NOTE: no way to use a mask to properly cut on matched track info
            self.pass_delta_eta = self._fail_ignore
        else:
            self.pass_delta_eta = self._pass_delta_eta
    @property
    def delta_phi(self):
        return self._delta_phi
    @delta_phi.setter
    def delta_phi(self, delta_phi):
        self._delta_phi = self._canonicalize('delta_phi', delta_phi)
        if self._delta_phi is None:
            # NOTE: no way to use a mask to properly cut on matched track info
            self.pass_delta_phi = self._fail_ignore
        else:
            self.pass_delta_phi = self._pass_delta_phi
    @property
    def ep_max(self):
        return self._ep_max
    @ep_max.setter
    def ep_max(self, ep_max):
        self._ep_max = self._canonicalize('ep_max', ep_max)
        if self._ep_max is None:
            # NOTE: no way to use a mask to properly cut on matched track info
            self.pass_ep_max = self._fail_ignore
        else:
            self.pass_ep_max = self._pass_ep_max

    def apply_to(self, df, label = "selected"):
        self.logger.warning("Cuts related to matched tracks cannot be properly applied to RDFs, so any cuts specified will be ignored.")
        if 'cluster_pt' not in df.GetColumnNames():
            df_masked =  df.Define("cluster_pt", "cluster_energy / cosh(cluster_eta)")
        else:
            df_masked = df
        df_masked = df_masked.Define("cluster_eta_abs", "abs(cluster_eta)")
        mask = self.rdf_mask
        self.logger.info(f"Applying cluster mask: {(mask)}")
        df_masked = df_masked.Define("cluster_mask", mask)
        cluster_cols = [col for col in df_masked.GetColumnNames() if col.startswith("cluster_")]
        name_map = {col: f"{col}_{label}" for col in cluster_cols}
        for colname, masked_colname in name_map.items():
            df_masked = df_masked.Define(masked_colname, f"{colname}[cluster_mask]")
        return df_masked

    def _pass_energy_min(self, cluster):
        """Checks if the cluster has enough energy."""
        return cluster.energy() >= self._energy_min
    def _pass_pt_min(self, cluster):
        """Checks if the cluster has large enough pT."""
        return cluster.pt() >= self._pt_min
    def _pass_m02_min(self, cluster):
        return cluster.m02() >= self._m02_min
    def _pass_m02_max(self, cluster):
        return cluster.m02() <= self._m02_max
    def _pass_ncells_min(self, cluster):
        """Checks if the cluster has at least the minimum number of cells."""
        return cluster.ncells() >= self._ncells_min
    def _pass_time_min(self, cluster):
        return cluster.time() >= self._time_min
    def _pass_time_max(self, cluster):
        return cluster.time() <= self._time_max
    # Special handling for the exoticity cut, which is a boolean. If exoticity is
    # set to True, then the selector will select non-exotic clusters, and vice versa.
    # If set to None, then the cluster exoticity is ignored during selection.
    def _pass_exotic(self, cluster):
        """Checks if the cluster passes the exoticity cut (i.e. it is non-exotic). Used if exoticity = True."""
        return not cluster.exoticity()
    def _fail_exotic(self, cluster):
        """Checks if the cluster does not pass the exoticity cut (i.e. it is exotic). Used if exoticity = False."""
        return cluster.exoticity()
    def _pass_dbc(self, cluster):
        """Checks if the cluster is far enough away from a bad EMC channel."""
        return cluster.dbc() >= self._dbc
    def _pass_nlm(self, cluster):
        """Checks if the cluster has few enough local maxima."""
        return cluster.nlm() <= self._nlm
    def _pass_defn(self, cluster):
        """Checks the cluster definition."""
        return cluster.defn() == self._defn
    def _pass_edge(self, cluster):
        # TODO: check if this cut is done right
        """Checks if the cluster is far enough away from a SM edge."""
        abs_eta = abs(cluster.eta())
        phi = cluster.phi()
        return (abs_eta < 0.67 and ( \
                    (phi > self.sm_full[0] and phi < self.sm_full[1]) or \
                    (phi > self.sm_13e[0] and phi < self.sm_13e[1]) or \
                    (phi > self.sm_13d[0] and phi < self.sm_13d[1]) )) \
            or (abs_eta > 0.25 and abs_eta < 0.67 and phi > self.sm_23[0] and phi < self.sm_23[1])
    def pass_cluster_cuts(self, cluster):
        return self.pass_energy_min(cluster) and self.pass_pt_min(cluster) \
           and self.pass_m02_min(cluster)    and self.pass_m02_max(cluster) \
           and self.pass_ncells_min(cluster) and self.pass_time_min(cluster) \
           and self.pass_time_max(cluster)   and self.pass_exotic(cluster) \
           and self.pass_dbc(cluster)        and self.pass_nlm(cluster) \
           and self.pass_defn(cluster)       and self.pass_edge(cluster)

    def _pass_delta_eta(self, cluster, idx):
        """Checks if a cluster and the given matched track match in eta."""
        return abs(cluster.matchedTrackDeltaEta()[idx]) <= self._delta_eta
    def _pass_delta_phi(self, cluster, idx):
        """Checks if a cluster and the given matched track match in phi."""
        return abs(cluster.matchedTrackDeltaPhi()[idx]) <= self._delta_phi
    def _pass_ep_max(self, cluster, idx):
        """Checks if a cluster and track match in energy."""
        return cluster.energy() / cluster.matchedTrackP()[idx] <= self._ep_max
    def is_not_in_sector_edge(self, cluster, idx):
        # TODO: implement TPC sector edge cut from LF
        return True
    def is_matched_to_track(self, cluster, idx):
        """Checks if a cluster and the given matched track match."""
        return self.pass_delta_eta(cluster, idx) and self.pass_delta_phi(cluster, idx) \
           and self.pass_ep_max(cluster, idx) and self.is_not_in_sector_edge(cluster, idx)
    def is_matched(self, cluster):
        return any(self.is_matched_to_track(cluster, idx) for idx in range(cluster.matchedTrackN()))
    def selects(self, cluster):
        """Determine if cluster is a photon candidate and is not matched to any track."""
        return self.pass_cluster_cuts(cluster) and not self.is_matched(cluster)
    def __call__(self, cluster):
        return self.selects(cluster)

class IsolationSelector(Selector):
    """An cluster selector for isolation, configured via YAML.

    The IsolationSelector contains the isolation criteria for EMCal clusters. To ignore the isolation pT
    altogether, set to None or leave unspecified in the config. Unlike other selections, `iso_cone_R` and
    `track_eta_max` will be set to default values of 0.4 and 0.9 if None or unspecified in the config.

    Attributes:
        iso_cone_R (float/int): isolation cone radius. If None or unspecified, this is set to 0.4.
        iso_pt_max (float/int): maximum isolation transverse momentum to be considered isolated.
        track_eta_max (float/int): maximum pseudorapidity for tracks. Only relevant for the geometric acceptance correction. If None or unspecified, this is set to 0.9.
    """

    _types = {
        # isolation cuts
        "iso_cone_R": (float, int), # isocone radius
        "iso_pt_max": (float, int), # max isocone pT (GeV)
        "track_eta_max": (float, int), # maximum track pseudorapidity
    }
    _name = "isolation"

    @property
    def iso_pt_max(self):
        return self._iso_pt_max
    @iso_pt_max.setter
    def iso_pt_max(self, iso_pt_max):
        self._iso_pt_max = self._canonicalize('iso_pt_max', iso_pt_max)
        # IsolationSelector only checks the iso pT, so we define selects() directly here
        if self._iso_pt_max is None:
            self.selects = self._pass_ignore
        else:
            self.selects = self._pass_iso_pt_max
    @property
    def iso_cone_R(self):
        return self._iso_cone_R
    @iso_cone_R.setter
    def iso_cone_R(self, iso_cone_R):
        self._iso_cone_R = self._canonicalize('iso_cone_R', iso_cone_R)
        if self._iso_cone_R is None: # None value not allowed in this case, so set to 0.4
            self.logger.warning("Isolation cone radius unspecified; setting to 0.4.")
            self._iso_cone_R = 0.4
    @property
    def track_eta_max(self):
        return self._track_eta_max
    @track_eta_max.setter
    def track_eta_max(self, track_eta_max):
        self._track_eta_max = self._canonicalize('track_eta_max', track_eta_max)
        if self._track_eta_max is None: # None value not allowed in this case, so set to 0.9
            self.logger.warning("Maximum track eta unspecified; setting to 0.9.")
            self._track_eta_max = 0.9

    def _pass_iso_pt_max(self, cluster, tracks, verbose = False):
        """Determine if cluster is isolated."""
        iso_cone_pt = np.sum([t.pt() for t in tracks if cluster.dR(t) < self._iso_cone_R])
        facc = self._facc(cluster.eta())
        iso_cone_pt_eff = iso_cone_pt * facc
        if not verbose:
            return iso_cone_pt_eff < self._iso_pt_max
        else:
            return iso_cone_pt_eff < self._iso_pt_max, iso_cone_pt_eff, facc, iso_cone_pt

    def apply_to(self, df):
        raise NotImplementedError("Isolation criteria can't be applied to RDataFrames!")

    def _facc(self, eta):
        """Calculate the geometric acceptance correction for clusters whose isolation cones extend outside the central barrel acceptance."""
        eta = abs(eta)
        if eta < (self._track_eta_max - self._iso_cone_R):
            return 1.0
        else:
            return np.pi * self._iso_cone_R * self._iso_cone_R / ((self._track_eta_max - eta) * np.sqrt(self._iso_cone_R**2 - (self._track_eta_max - eta)**2 ) + np.pi * self._iso_cone_R**2 * (1 - 1 / np.pi * np.arccos((self._track_eta_max - eta) / self._iso_cone_R)))

class AnalysisSelector:
    def __init__(self, file):
        self.logger = set_up_logger(type(self).__name__)
        self.event = EventSelector.load(file)
        self.track = TrackSelector.load(file)
        self.cluster = ClusterSelector.load(file)
        self.isolation = IsolationSelector.load(file)

    @classmethod
    def load(cls, file):
        return cls(file)

    def dump(self):
        self.logger.info("Selector configurations:")
        for selector in self.active:
            getattr(self, selector).dump()
        self.logger.info(f"Inactive selectors: {self.inactive}")

    @property
    def active(self):
        """Return a list of active selectors."""
        selectors = ["event", "track", "cluster", "isolation"]
        return [selector for selector in selectors if self.is_active(selector)]

    @property
    def inactive(self):
        """Return a list of inactive selectors."""
        selectors = ["event", "track", "cluster", "isolation"]
        return [selector for selector in selectors if self.is_inactive(selector)]

    def is_active(self, selector: str):
        return getattr(self, selector) is not None
    def is_inactive(self, selector: str):
        return getattr(self, selector) is None