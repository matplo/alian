from time import perf_counter

from alian.io import data_io
from ROOT import TFile

import heppyy

from .event import Event, get_selected_clusters, get_selected_tracks
from .jet_finder import JetFinder
from .logs import set_up_logger
from .output import Output
from .selector import AnalysisSelector
from .utils import is_slurm, read_yaml


class BaseAnalysis(heppyy.GenericObject):
    _defaults = {}

    def __init__(self, **kwargs):
        super(BaseAnalysis, self).__init__(**kwargs)
        for k, val in self.__class__._defaults.items():
            if not hasattr(self, k) or getattr(self, k) is None:
                setattr(self, k, val)

class AnalysisBase:
    """A base class for analysis tasks.

    This base class describes the basic event analysis structure.
    At minimum, the method `analyze_event()` must be overridden. Depending
    on the selectors defined in the configuration, clusters may or may not be
    loaded into the event.
    """
    _defaults = {}

    def __init__(self, input_file, output_file, cfg, tree_struct, nev = -1, lhc_run = 3):
        self.input_file = input_file
        self.output_file = output_file
        self.cfg_file = cfg
        self.tree_struct = tree_struct
        self.nev = nev
        self.lhc_run = lhc_run

    def run(self):
        self.start_time = perf_counter()
        # set up logger
        self.logger = set_up_logger(__name__)
        self.logger.info("Starting analysis!")

        self.logger.info("Configuring analysis pipeline...")
        self.init()
        self.logger.info("Analysis pipeline configured.")
        self.analyze_events()
        self.finalize()
        self.save()

        self.note_time("Analysis complete")

    def init(self):
        self.cfg = read_yaml(self.cfg_file)
        # validate YAML config
        self._validate_cfg(self.cfg)
        self.logger.info("Config schema validated.")
        # initialize data input
        self.data_source = data_io.DataInput(self.input_file,
                                             lhc_run = self.lhc_run,
                                             yaml_file = self.tree_struct,
                                             n_events = self.nev)
        # initialize selectors for events, tracks, clusters
        self.logger.info("Configuring selectors...")
        self.selector = AnalysisSelector.load(self.cfg)
        self.selector.dump()
        self.logger.info("Selectors configured.")
        if self.selector.is_active("cluster"):
            self.logger.info("Cluster selector found, clusters will be loaded into events.")
            self.load_clusters = True
        else:
            self.logger.info("Cluster selector not found, clusters will NOT be loaded into events.")
            self.load_clusters = False
        # initialize jet finder
        if 'jet_finder' in self.cfg:
            self.logger.info("Jet finder config found, jets will be loaded into events.")
            self.load_jets = True
            self.logger.info("Configuring jet finder...")
            self.jet_finder = JetFinder.load(self.cfg)
            self.jet_finder.dump()
            self.logger.info("Jet finder configured.")
        else:
            self.logger.info("No configuration for jet finder, no jets will be loaded.")
            self.load_jets = False
        self.logger.info("Configuring output...")
        self.init_output()
        self.logger.info("Output configured.")
        # initialize analysis parameters
        self.logger.info("Configuring analysis parameters...")
        self.init_analysis(self.cfg.get('analysis', {}))
        self.dump()
        self.logger.info("Analysis parameters configured.")

    def _validate_cfg(self, cfg: dict):
        req_headers = ['selections', 'output']
        if headers_not_in_cfg := [h for h in req_headers if h not in cfg]:
            raise KeyError(f"The following blocks must be present in the YAML configuration: {headers_not_in_cfg}")

    def init_analysis(self, analysis_cfg: dict):
        """Initialize analysis configuration parameters. Override in analyses if necessary."""
        self.logger.info("No analysis configuration to parse.")

    def init_output(self):
        self.output = Output.load(self.cfg)
        self.hists = self.output.hists
        self.trees = self.output.trees

    def analyze_events(self):
        self.logger.info("Analyzing events...")
        slurm_check = is_slurm()
        for e in self.data_source.next_event(disable_bar = slurm_check):
            # build the event only
            self.build_event(e)
            # skip if the event doesn't pass the selection
            if not self.selector.event.selects(self.event):
                continue
            # build the rest of the event and analyze
            self.build_event_objs(e)
            self.analyze_event()
        self.note_time("Events analyzed")

    def analyze_event(self):
        raise NotImplementedError("Must implement an analyze_event method!")

    def build_event(self, event_struct):
        """Build event as Event and store in self.event."""
        self.event = Event(event_struct)
    def build_event_objs(self, event_struct):
        """Build tracks and optionally clusters and jets from associated event."""
        self.tracks = get_selected_tracks(event_struct, self.selector.track)
        if self.load_clusters:
            self.clusters = get_selected_clusters(event_struct, self.selector.cluster)
        if self.load_jets:
            self.jets = self.jet_finder.find_jets(self.tracks)

    def finalize(self):
        """Finalize analysis before saving output. Should be overriden in analyses if necessary."""
        self.logger.info("Nothing to finalize.")
    def save(self):
        self.logger.info(f"Saving output to: {self.output_file}")
        with TFile(self.output_file, "RECREATE") as f:
            self.output.save(f)
        self.logger.info("Output saved.")
    def __getattr__(self, attr):
        if attr == "jets":
            raise AttributeError("Jets not defined; include a `jet_finder` block in your config!")
        if attr == "clusters":
            raise AttributeError("Clusters not defined; include a `clusters` block in the `selections` section of your config! The block can be empty if you want no selections applied.")
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def dump(self):
        cfg = "\n".join([f"\t{param}: {repr(getattr(self, param))}" for param in self._defaults])
        self.logger.info(f"{type(self).__name__} configuration:\n{cfg}", stacklevel = 2)
    def note_time(self, msg):
        self.logger.info(f"{msg}: ------- {self.fmt_time(perf_counter() - self.start_time)} -------", stacklevel = 2)
    def fmt_time(self, seconds):
        m, s = divmod(round(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}h {m:02d}m {s:02d}s"

def add_default_args(parser):
    parser.add_argument('-i', '--input-file',  type = str, required = True,           help = "Input file or file containing a list of input files.")
    parser.add_argument("-o", "--output-file", type = str, default = "analysis.root", help = "File to write the analysis.")
    parser.add_argument('-c', '--config-file', type = str, required = True,           help = "Path to a YAML configuration file describing the cuts to be applied to the data.")
    parser.add_argument("-n", "--nev",         type = int, default = -1,              help = "Number of entries to process, set to -1 for all events.")
    parser.add_argument('-t', '--tree-struct', type = str, default = None,            help = "Path to a YAML file describing the tree structure.")
    parser.add_argument('--lhc-run',           type = int, default = 3,               help = 'LHC Run')

    return parser
