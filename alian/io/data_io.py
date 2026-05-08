#!/usr/bin/env python

import os
import uproot
import yaml
import pandas as pd
import yasp
from alian.steer.tqdm import tqdm

class Run3FileInput(yasp.GenericObject):
    def __init__(self, file_list, yaml_file=None, tree_name=None, branches=None, **kwargs):
        super(Run3FileInput, self).__init__(**kwargs)
        self.file_list = file_list
        if isinstance(self.file_list, str):
            if self.file_list.endswith(".txt"):
                with open(self.file_list, "r") as file:
                    self.file_list = file.read().splitlines()
            else:
                self.file_list = [self.file_list]
        self.tree_name = tree_name
        self.branches = branches
        if yaml_file is not None:
            with open(yaml_file, 'r') as file:
                tree_structure = yaml.safe_load(file)
            for tname, tinfo in tree_structure.items():
                self.tree_name = tname
                self.branches = tinfo['branches']
        if self.name is None:
            self.name = "FileIORun3"
        self.event = None
        self.event_count = 0
        self.n_events = kwargs.get('n_events', -1)

    def add_generic_ebye_info(self):
        self.event.multiplicity = self.event.data['multiplicity']
        self.event.centrality = self.event.data['centrality']
        self.event.track_count = len(self.event.data['track_pt'])
        self.event.counter = self.event_count

    # Efficiently iterate over the tree as a generator
    def next_event(self, step_size=4000, disable_bar = False, smoothing = 0.3):
        # Count total number of events
        total_events = 0
        events_per_file = []
        for root_file_path in self.file_list:
            # Count total number of events over all files
            with uproot.open(f"{root_file_path}:{self.tree_name}") as tree:
                events_per_file.append(tree.num_entries)
        total_events = sum(events_per_file)
        if self.n_events < 0:
            # if number of events unspecified, read all events
            self.n_events = total_events
        else:
            # if number of events specified, read up to given number
            self.n_events = min(self.n_events, total_events)

        # Bar for number of files processed
        pbar_files = tqdm(total = len(self.file_list), desc="Files", unit = "files", position = 0, disable = disable_bar, smoothing = smoothing)
        # Bar for all events across all files
        pbar_total = tqdm(total = self.n_events, desc="Total events", unit = "ev", position = 1, disable = disable_bar, smoothing = smoothing)
        # Bar for events within each file
        pbars = [tqdm(total=nev, desc=f'...{root_file_path[-20:]}', unit = "ev", position = ifile+2, disable = disable_bar, smoothing = smoothing)
                 for ifile, (nev, root_file_path) in enumerate(zip(events_per_file, self.file_list))
                 ]
        self.event = yasp.GenericObject()
        for pbar, root_file_path in zip(pbars, self.file_list):
            # Open the ROOT file and access the tree
            with uproot.open(f"{root_file_path}:{self.tree_name}") as tree:
                for batch in tree.iterate(self.branches, library="np", step_size=step_size):
                    # Iterate over the events in the chunk
                    for event_data in zip(*batch.values()):
                        # event_data is a tuple of 1-D ndarrays for one event, one from each branch
                        self.event.data = {branch: branch_data for branch, branch_data in zip(batch.keys(), event_data)}
                        self.add_generic_ebye_info()
                        pbar.update(1)
                        pbar_total.update(1)
                        yield self.event
                        self.event_count += 1
                        if self.event_count >= self.n_events:
                            pbar_files.close()
                            pbar_total.close()
                            [bar.close() for bar in pbars]
                            return
            pbar.refresh()
            pbar_files.update(1)
        pbar_files.close()
        pbar_total.close()
        [bar.close() for bar in pbars]


class Run2FileInput(yasp.GenericObject):
    def __init__(self, file_list, yaml_file, **kwargs):
        super(Run2FileInput, self).__init__(**kwargs)
        self.file_list = file_list
        if isinstance(self.file_list, str):
            if self.file_list.endswith(".txt"):
                with open(self.file_list, "r") as file:
                    self.file_list = file.read().splitlines()
            else:
                self.file_list = [self.file_list]
        self.setup_default_trees_and_branches(yaml_file)
        if self.name is None:
            self.name = "FileIORun2"
        self.event = None
        self.event_count = 0

    def setup_default_trees_and_branches(self, yaml_file_path):
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)

        self.tree_particle_name = 'PWGHF_TreeCreator/tree_Particle'
        self.tree_event_char_name = 'PWGHF_TreeCreator/tree_event_char'

        self.branches_particle = list(config['PWGHF_TreeCreator']['tree_Particle'].keys())
        self.branches_event_char = list(config['PWGHF_TreeCreator']['tree_event_char'].keys())

        self.branches = self.branches_particle + self.branches_event_char

    def add_generic_ebye_info(self):
        self.event.multiplicity = list(set(self.event.data['V0Amult']))[0]
        self.event.centrality = list(set(self.event.data['centrality']))[0]
        self.event.track_count = len(self.event.data['ParticlePt'])
        self.event.counter = self.event_count

    # Efficiently iterate over the tree as a generator
    def next_event(self):
        self.event = None
        pbar_total = None
        self.event = yasp.GenericObject()
        self.event.lhc_run = 2
        if self.n_events > 0:
            pbar_total = tqdm(total=self.n_events, desc="Total events")
        for root_file_path in tqdm(self.file_list, desc="Files"):
            # Open the ROOT file
            file = uproot.open(root_file_path)
            # Access the trees
            tree_particle = file[self.tree_particle_name]
            tree_event_char = file[self.tree_event_char_name]

            # Convert the trees to pandas DataFrames
            df_particle = tree_particle.arrays(library="pd")
            df_event_char = tree_event_char.arrays(library="pd")

            # Merge the DataFrames on 'ev_id', 'ev_id_ext', and 'run_number'
            data_merge_on = ['ev_id', 'ev_id_ext', 'run_number']
            # check if ev_id_ext present in both DataFrames
            if 'ev_id_ext' not in df_particle.columns or 'ev_id_ext' not in df_event_char.columns:
                data_merge_on.remove('ev_id_ext')
            merged_df = pd.merge(df_particle, df_event_char, on=data_merge_on)

            # Filter out events where 'is_ev_rej' is not zero and 'centrality' is not less than 10
            filtered_df = merged_df[(merged_df['is_ev_rej'] == 0) & (merged_df['centrality'] < 10)]

            # Group by 'ev_id', 'ev_id_ext', and 'run_number'
            grouped_df = filtered_df.groupby(data_merge_on)

            # Efficiently iterate over the grouped DataFrame with a progress bar
            total_entries = len(grouped_df)
            with tqdm(total=total_entries, desc=f'...{root_file_path[-25:]}') as pbar:
                for name, group in grouped_df:
                    self.event.data = {branch: group[branch].tolist() for branch in self.branches if branch in group}
                    self.add_generic_ebye_info()
                    pbar.update(1)
                    if pbar.n >= total_entries:
                        break
                    self.event_count += 1
                    if pbar_total is not None:
                        pbar_total.update(1)
                        if pbar_total.n >= self.n_events:
                            break
                    yield self.event
            if pbar_total is not None:
                if pbar_total.n >= self.n_events:
                    pbar_total.close()
                    break


def get_default_tree_structure(file_list, lhc_run=3):
    if isinstance(file_list, str):
        if file_list.endswith(".txt"):
            with open(file_list, "r") as file:
                filename = file.read().splitlines()
        else:
            filename = file_list
    if os.path.isfile(_guess_path := os.path.join(os.path.dirname(os.path.abspath(filename)), "../tstruct.yaml")):
        return _guess_path
    _alian_path_guess = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    _default_tree_structure = os.path.join(_alian_path_guess, f'config/run{lhc_run}_tstruct.yaml')
    _alian_path = yasp.features('prefix', 'alian')
    if len(_alian_path) > 0:
        _alian_path = _alian_path[0]
        _default_tree_structure = os.path.join(_alian_path, f'lib/alian/config/run{lhc_run}_tstruct.yaml')
    if not os.path.exists(_default_tree_structure):
        print(f"Default tree structure file {_default_tree_structure} not found.")
    return _default_tree_structure

class DataInput(yasp.GenericObject):
    def __init__(self, file_list, lhc_run=None, yaml_file=None, tree_name=None, branches=None, **kwargs):
        super(DataInput, self).__init__(**kwargs)
        self.file_list = file_list
        self.lhc_run = lhc_run
        self.yaml_file = yaml_file
        self.tree_name = tree_name
        self.branches = branches
        if self.yaml_file is None:
            self.yaml_file = get_default_tree_structure(file_list = file_list, lhc_run=self.lhc_run)
        _data_input = Run2FileInput if lhc_run == 2 else Run3FileInput
        self.data_input = _data_input(file_list, yaml_file=self.yaml_file, tree_name=self.tree_name, branches=self.branches, **kwargs)

    def next_event(self, *args, **kwargs):
        return self.data_input.next_event(*args, **kwargs)
