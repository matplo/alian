import heppyy.util.fastjet_cppyy
from cppyy.gbl import fastjet as fj
from cppyy.gbl import std

import heppyy

from .selection import EventSel, RCTSel, TrigSel

alian = heppyy.load_cppyy("alian")

class Event:
    def __init__(self, ev):
        self.run_number = ev.data["run_number"]
        self.multiplicity = ev.data["multiplicity"]
        self.centrality = ev.data["centrality"]
        self.occupancy = ev.data["occupancy"]
        self.event_sel = EventSel(ev.data["event_sel"])
        self.trig_sel = TrigSel(ev.data["trig_sel"])
        self.rct = RCTSel(ev.data["rct"])


def get_tracks(ev):
    """Get all tracks from an event (no track selections applied)."""
    return alian.numpy_ptetaphi_to_tracks(
        ev.data["track_pt"],
        ev.data["track_eta"],
        ev.data["track_phi"],
        ev.data["track_sel"],
        0,
    )


def get_selected_tracks(ev, selector):
    """Get selected tracks from an event (track selections applied)."""
    return std.vector[fj.PseudoJet](
        [
            t
            for t in alian.numpy_ptetaphi_to_tracks(
                ev.data["track_pt"],
                ev.data["track_eta"],
                ev.data["track_phi"],
                ev.data["track_sel"],
                0,
            )
            if selector.selects(t)
        ]
    )


def get_clusters(ev):
    """Get all clusters from an event (no cluster selections applied)."""
    return alian.numpy_energyetaphi_to_clusters(
        ev.data["cluster_energy"],
        ev.data["cluster_eta"],
        ev.data["cluster_phi"],
        ev.data["cluster_m02"],
        ev.data["cluster_m20"],
        ev.data["cluster_ncells"],
        ev.data["cluster_time"],
        ev.data["cluster_exoticity"],
        ev.data["cluster_dbc"],
        ev.data["cluster_nlm"],
        ev.data["cluster_defn"],
        ev.data["cluster_matched_track_n"],
        ev.data["cluster_matched_track_delta_eta"],
        ev.data["cluster_matched_track_delta_phi"],
        ev.data["cluster_matched_track_p"],
        ev.data["cluster_matched_track_pt"],
        ev.data["cluster_matched_track_sel"],
        0,
    )


def get_selected_clusters(ev, selector):
    """Get selected clusters from an event (cluster selections applied)."""
    return [
        c
        for c in alian.numpy_energyetaphi_to_clusters(
            ev.data["cluster_energy"],
            ev.data["cluster_eta"],
            ev.data["cluster_phi"],
            ev.data["cluster_m02"],
            ev.data["cluster_m20"],
            ev.data["cluster_ncells"],
            ev.data["cluster_time"],
            ev.data["cluster_exoticity"],
            ev.data["cluster_dbc"],
            ev.data["cluster_nlm"],
            ev.data["cluster_defn"],
            ev.data["cluster_matched_track_n"],
            ev.data["cluster_matched_track_delta_eta"],
            ev.data["cluster_matched_track_delta_phi"],
            ev.data["cluster_matched_track_p"],
            ev.data["cluster_matched_track_pt"],
            ev.data["cluster_matched_track_sel"],
            0,
        )
        if selector.selects(c)
    ]
