from pathlib import Path
from params import datadir
import pickle
import pandas as pd
import neo
import numpy as np
import operator
from files_from_neo_09dev.nixio_fr import NixIO


class Io:

    # When a session is loaded, these attributes are updated
    animal_id = None  # Id of animal
    session_date = None  # Date of session
    session_id = None  # Id of session
    data_version = None  # Version of current dataset
    block = neo.Block()  # Neo block containing the data

    unit_ids = list()  # all unit ids in session
    unit_quality_criteria = None  # criteria for assigning a unit is 'good' or 'bad'
    unit_celltype_criteria = None  # criteria for assigning 'celltype' to a unit
    unit_metanames = None  # metainformation annotated on each unit

    trial_ids = list()  # all trial ids in session
    trial_metanames = None  # names of meta information per trial
    noise_trial_ids = list()

    lfp_ids = list()
    lfp_clean_ids = list()
    lfp_metanames = list()
    lfp_clean_metanames = list()

    eventnames = None  # names of available events

    trial_df = pd.DataFrame()  # dataframe with trial metainformation
    unit_df = pd.DataFrame()  # dataframe with unit meta information
    lfp_df = pd.DataFrame()
    lfp_clean_df = pd.DataFrame()

    sessions_df = None
    debug = None  # allow to do weird stuff if True

    def __init__(self, path=datadir, sid=None, read_lfp=False):
        """

        Parameters
        ----------
        path
            Directory wherefrom to read the dataset (don't pass to read default directory)
        read_lfp
            Whether the io should also read to lfp files (faster if not). read_lfp can also
            be passed to load_session
        """

        # Whole io now works with pathlib, check path datatype and check if path exists
        if not isinstance(path, Path):
            path = Path(path)

        if not path.is_dir():
            raise ValueError(f'path {path}" does not exist')

        self.read_lfp = read_lfp  # Wheter to read lfp files or not during 'load_session' call
        self.dataset_path = path  # Path containing the data

        # list all available session ids
        self.session_ids = [i.name.split('neo_')[1].split('.pkl')[0] for i in self.dataset_path.iterdir() if
                            'neo_rat' in i.name and '.pkl' in i.name]
        assert len(self.session_ids) > 0, f'no sessinos found in : {path}'

        # List animals
        self.animal_ids = []

        for sid in self.session_ids:
            animal = sid.split('_')[0]
            if animal not in self.animal_ids:
                self.animal_ids.append(animal)

        sessions_df_name = path / 'sessions_df.csv'
        if not sessions_df_name.is_file():
            self._get_sessions_df(sessions_df_name)

        self.sessions_df = pd.read_csv(sessions_df_name.as_posix(), header=0, index_col=0)


    # -----------------------------------------------------------------------------------
    #                           METHODS FOR LOADING DATA
    # -----------------------------------------------------------------------------------


    def load_session(self, session, read_lfp=None, debug=False):
        # Handle spikefield reading state
        if read_lfp is not None:
            self.read_lfp = read_lfp

        # if session is an integer, load the nth session in the dataset
        if isinstance(session, int):
            session = self.session_ids[session]

        # Extract ids
        self.session_id = session
        self.animal_id, self.session_date = session.split('_')

        # Load session neo block
        with open((self.dataset_path / f'neo_{self.session_id}.pkl'), 'rb') as f:
            self.block = pickle.load(f)

        if self.read_lfp:
            self._load_lfp()

        # self._annotate_recording_group()

        self._update_attributes()  # Sets the attributes for this session
        self._get_dataframes(debug)  # Store  meta infomration in pandas

    def _update_attributes(self):
        """
        Update class atributes for currently loaded session
        """
        # Unit attributes
        self.unit_ids = [sp.name for sp in self.block.segments[0].spiketrains]
        self.unit_quality_criteria = self.block.annotations['unit_criteria']
        self.unit_celltype_criteria = self.block.annotations['unit_classification_thresholds']
        if 'unit_metanames' in self.block.annotations.keys():
            self.unit_metanames = self.block.annotations['unit_metanames']
        # self.unit_metanames.append('recording_group')

        # Trial attributes
        trial_meta = self.get_object('trial_meta')
        self.trial_ids = [l for l in trial_meta.labels if 'xxx' not in l]
        self.trial_metanames = self.block.annotations['trial_metanames']
        self.eventnames = self.block.annotations['eventnames']

        # spikefield attributes
        lfp_df = self.block.annotations['lfpframe']
        lfp_metanames = []
        for lfp_id in lfp_df.keys():
            for name in lfp_df[lfp_id].keys():
                if name not in lfp_metanames:
                    lfp_metanames.append(name)
        self.lfp_metanames = lfp_metanames

        self.lfp_clean_metanames = []
        for c in self.lfp_clean_df.columns:
            self.lfp_clean_metanames.append(c)

        self._find_noise_trials()

    def _get_dataframes(self, debug=False):
        """
        Update class dataframe attributes, for trials, units and lfps
        """

        # Get units dataframe
        names_to_skip = ['waveform_mean', 'waveform_std', 'template', 'location_notes',
                         'excluding_criteria']
        self.unit_df = pd.DataFrame()
        for sp in self.block.segments[0].spiketrains:
            uid = sp.annotations['unitid']
            for name in self.unit_metanames:
                if name in names_to_skip:
                    continue
                self.unit_df.at[uid, name] = sp.annotations[name]

        # Get trials dataframe
        trial_meta = self.get_object('trial_meta')
        self.trial_df = pd.DataFrame()
        for i, tid in enumerate(trial_meta.labels):
            if 'xxx' in tid:
                continue

            for name in self.trial_metanames:
                self.trial_df.at[tid, name] = trial_meta.array_annotations_at_index(i)[name]

        # Get lfp dataframe
        self.lfp_df = pd.DataFrame()
        lfpmeta = self.block.annotations['lfpframe']
        self.lfp_ids = []
        for lfp_id in lfpmeta.keys():
            self.lfp_ids.append(lfp_id)
            for name in lfpmeta[lfp_id].keys():
                self.lfp_df.at[lfp_id, name] = lfpmeta[lfp_id][name]

            # recording_group = None
            # if self.animal_id in cfg.recording_groups.keys():
            #     for g, i in cfg.recording_groups[self.animal_id].items():
            #         if self.lfp_df.at[lfp_id, 'tetrode_area'] in i:
            #             recording_group = g
            #
            #     if recording_group is None:
            #         recording_group = self.lfp_df.at[lfp_id, 'tetrode_area']
            #     self.lfp_df.at[lfp_id, 'recording_group'] = recording_group
            # else:
                # self.lfp_df.at[lfp_id, 'recording_group'] = self.lfp_df.at[lfp_id, 'tetrode_area']
                # print('skipped tetrode_area')

        # if not debug:
        #     assert 'lfpframe_clean_dict' in self.block.annotations.keys(), 'run get_clean_lfps for this session'

        if 'lfpframe_clean_dict' in self.block.annotations.keys():
            lfpmeta = self.block.annotations['lfpframe_clean_dict']
            self.lfp_clean_df = pd.DataFrame()
            self.lfp_clean_ids = []
            for lfp_id in lfpmeta.keys():
                self.lfp_clean_ids.append(lfp_id)
                for name in lfpmeta[lfp_id].keys():
                    self.lfp_clean_df.at[lfp_id, name] = lfpmeta[lfp_id][name]

                # recording_group = None
                # if self.animal_id in cfg.recording_groups.keys():
                #     for g, i in cfg.recording_groups[self.animal_id].items():
                #         if self.lfp_clean_df.at[lfp_id, 'tetrode_area'] in i:
                #             recording_group = g
                #
                #     if recording_group is None:
                #         recording_group = self.lfp_clean_df.at[lfp_id, 'tetrode_area']
                #     self.lfp_clean_df.at[lfp_id, 'recording_group'] = recording_group
                # else:
                #     self.lfp_clean_df.at[lfp_id, 'recording_group'] = self.lfp_clean_df.at[lfp_id, 'tetrode_area']

    def _get_sessions_df(self, savename):
        df = pd.DataFrame()
        for sid in self.session_ids:
            animal = sid.split('_')[0]
            date = sid.split('_')[1]

            df.at[sid, 'animal'] = animal
            df.at[sid, 'date'] = date

        df.to_csv(savename.as_posix())



    def _load_lfp(self):
        """
        Add lfp metainfomration to block
        Returns
        -------

        """
        lfpfile = self.dataset_path / f'neo_{self.session_id}_lfps.nio'
        if lfpfile.is_file():
            reader = NixIO(filename=lfpfile.as_posix())
            block = reader.read_block(lazy=True)

            lfpsig = block.segments[0].analogsignals[0]
            lfpsig.name = 'lfps'
            for i in range(lfpsig.shape[1]):
                lfpsig.array_annotations['channel_names'][i] = block.annotations['channel_names'][i]
                lfpsig.array_annotations['channel_ids'][i] = block.annotations['channel_ids'][i]
            self.block.segments[0].analogsignals.append(lfpsig)

        lfpfile_clean = self.dataset_path / f'neo_{self.session_id}_lfps_clean.nio'
        if lfpfile_clean.is_file():
            reader = NixIO(filename=lfpfile_clean.as_posix())
            block = reader.read_block(lazy=True)
            lfpsig = block.segments[0].analogsignals[0]
            lfpsig.name = 'lfps_clean'
            for i in range(lfpsig.shape[1]):
                lfpsig.array_annotations['channel_names'][0] = block.annotations['channel_names'][i]
            self.block.segments[0].analogsignals.append(lfpsig)

    # -----------------------------------------------------------------------------------
    #                           METHODS SELECTING DATA
    # -----------------------------------------------------------------------------------

    def select_trials(self, events_of_interest=None, no_noise_trials=False, **kwargs):
        if events_of_interest is not None:
            if not isinstance(events_of_interest, list):
                events_of_interest = [events_of_interest]
            for ev_name in events_of_interest:
                assert ev_name in self.eventnames, f'{ev_name} not in {self.eventnames}'

        trialids = []
        trial_meta = self.get_object('trial_meta')
        for i in range(len(trial_meta)):
            if 'xxxxx' in trial_meta.labels[i]:  # never return undefined trials
                continue

            valid_id = True  # Flag to keep or throw the trialid
            trial_metainfo = trial_meta.array_annotations_at_index(i)
            trial_id = trial_meta.labels[i]

            if events_of_interest is not None:
                for ev_name in events_of_interest:
                    evtime = self.get_event_times(event_name=ev_name, trial_ids=trial_id,
                                                  verbose=False)
                    if len(evtime) == 0:
                        valid_id = False

            if 'all' in kwargs.keys() and kwargs['all']:  # return all trial ids
                if valid_id:
                    trialids.append(trial_meta.labels[i])
                continue

            for meta_name in kwargs.keys():
                assert meta_name in trial_metainfo.keys(), f'{meta_name} not a meta name'
                if isinstance(kwargs[meta_name], list):
                    vals = kwargs[meta_name][:-1]
                    opr = kwargs[meta_name][-1]

                    keep_by_this_meta_name = False
                    for val in vals:
                        if self.get_object(opr)(trial_metainfo[meta_name], val):
                            keep_by_this_meta_name = True

                    if not keep_by_this_meta_name:
                        valid_id = False

                else:
                    val = kwargs[meta_name]
                    opr = '=='

                    # if isinstance(val, int):
                    #     val = f'{val:1.1f}'

                    if not self._get_operator(opr)(trial_metainfo[meta_name], val):
                        valid_id = False

            if no_noise_trials:
                if trial_id in self.noise_trial_ids:
                    valid_id = False
            if valid_id:
                trialids.append(trial_meta.labels[i])

        # assert len(trialids) > 1
        return trialids

    def select_units(self, **kwargs):
        """

        :param kwargs: colname=[condition, operator], ie: tetrode_area=['Barrel', '==']
        :return:
        """
        unitids = []
        for i in range(len(self.block.segments[0].spiketrains)):
            sp = self.block.segments[0].spiketrains[i]
            uid = sp.annotations['unitid']
            if 'all' in kwargs.keys() and kwargs['all']:
                unitids.append(uid)
                continue

            is_valid = True
            for metric in kwargs.keys():
                # assert metric in self.unit_metanames or metric in self.unit_metricnames, \
                #     f'{metric} not in {self.unit_metanames}, {self.unit_metricnames}'

                if isinstance(kwargs[metric], list):
                    vals = kwargs[metric][:-1]
                    opr = kwargs[metric][-1]

                    keep_by_metric_name = False
                    for val in vals:
                        if self._get_operator(opr)(sp.annotations[metric], val):
                            keep_by_metric_name = True

                    if not keep_by_metric_name:
                        is_valid = False

                else:
                    val = kwargs[metric]
                    opr = '=='

                    if not self._get_operator(opr)(sp.annotations[metric], val):
                        is_valid = False

            if is_valid:
                unitids.append(uid)
        return unitids

    def select_lfps(self, clean=False, **kwargs):
        if clean is True:
            raise ValueError('Clean LFPS not available in dataset...')
        lfpids = []
        if clean:
            df = self.lfp_clean_df
        else:
            df = self.lfp_df

        for i in df.index:
            if 'all' in kwargs.keys() and kwargs['all']:
                lfpids.append(i)
                continue

            is_valid = True
            for metric in kwargs.keys():
                assert metric in df.columns, f'{metric} not in {df.columns}'

                if isinstance(kwargs[metric], list):
                    vals = kwargs[metric][:-1]
                    opr = kwargs[metric[-1]]

                    keep_by_metric_name = False
                    for val in vals:
                        if self._get_operator(opr)(df.loc[i][metric], val):
                            keep_by_metric_name = True
                    if not keep_by_metric_name:
                        is_valid = False

                else:
                    val = kwargs[metric]
                    opr = '=='
                    if not self._get_operator(opr)(df.loc[i][metric], val):
                        is_valid = False
            if is_valid:
                lfpids.append(i)
        return lfpids

    # -----------------------------------------------------------------------------------
    #                           GET METHODS
    # -----------------------------------------------------------------------------------

    def get_event_times(self, trial_ids, event_name, verbose=True, return_ids=False):
        """
        Retrun event times, specified by trial ids and event name.
        If verbose is true, the method will report cases of trials where
        the event did not occur

        :param trial_ids: str or list of str
            trial ids to return event for
        :param event_name: str
            name of event to get time for
        :param verbose: bool
            if true, will report trials where event did not occur
        :param return_ids: bool
            if true, will return als the triald ids matching the times

        Output:
            quantities, list
                quantities array with even times and corresponding trial_ids

        """
        assert isinstance(event_name, str), 'event should be a string'
        assert event_name in self.eventnames, f'{event_name} not in {self.eventnames}'
        if not isinstance(trial_ids, list):
            trial_ids = [trial_ids]

        trial_meta = self.get_object('trial_meta')
        times_out, labels_out = [], []
        event = self.get_object(event_name)

        for tid in trial_ids:
            assert tid in trial_meta.labels, f'{tid} is not a valid trial id'
            idx = np.where(event.labels == tid)[0]
            if len(idx) == 0:
                if verbose:
                    print(f'{tid} not in {event_name}')
                continue

            for i in idx:
                times_out.append(event[i].magnitude)
                labels_out.append(tid)

        if not return_ids:
            return np.asarray(times_out) * event.units
        else:
            return np.asarray(times_out) * event.units, labels_out

    def get_aligned_times(self, trial_ids, event, time_before, time_after,
                          return_ids=False, verbose=False):
        """
        Get times in session, per selected trial, around en event of interest

        Note: if an event time for a given trial is None; it will not include the trial_is

        :param trial_ids: trial_ids: list of trial_ids (as in TrialFrame.index or .array_annotations['trialid'])
        :param event: event: string or neo.Event
        :param time_before: [quantities] time before event
        :param time_after: [quantities] time after event
        :param return_ids: [bool] if True, function returns trial ids
        :param verbose: [bool] if True, print errors
        :return: [neo.Event] times before, [neo.Event] times after
        """
        assert hasattr(time_before, 'units'), 'time_before should be quantities'
        assert hasattr(time_after, 'units'), 'time_after should be quantities'
        assert isinstance(event, str), 'eventname needs to be a string'

        event_times = self.get_event_times(trial_ids, event, verbose=verbose)
        ev_times = event_times.rescale('us')
        if return_ids:
            return ev_times - time_before, ev_times + time_after, trial_ids
        else:
            return ev_times - time_before, ev_times + time_after

    def get_object(self, object_id):
        """
        Return an object from the neo block based on its id. spikefield's treated separately as they
        are 'lazy' loaded.
        """
        if 'lfp' in object_id or 'CSC' in object_id:
            assert self.read_lfp, f'Set read lfp to true!'
            if 'clean' in object_id:
                channel_index = int(self.lfp_clean_df.loc[object_id].channel_index)
                lfps = self.block.filter(name='lfps_clean')[0]
                return lfps.load(time_slice=None, channel_indexes=[channel_index])
            else:
                channel_index = int(self.lfp_df.loc[object_id].channel_idx)
                lfps = self.block.filter(name='lfps')[0]
                return lfps.load(time_slice=None, channel_indexes=[channel_index])
        else:
            res = self.block.filter(name=object_id)
            if len(res) == 0:
                raise ValueError(f'{object_id} returned no objects')
            elif len(res) > 1:
                raise ValueError(f'{object_id} returned multiple objects')
            return res[0]

    def get_meta(self, object_id, name):
        if 'lfp' not in object_id and 'CSC' not in object_id and 'tid' not in object_id:
            obj = self.get_object(object_id)
            assert name in obj.annotations.keys(), f'{name} not in object annotations'
            return obj.annotations[name]

        elif 'tid' in object_id:
            assert object_id in self.trial_ids, f'{object_id} not a valid trial id (see io.trial_ids)'
            tm = self.get_object('trial_meta')
            trial_idx = [i for i in range(len(tm.labels)) if tm.labels[i] == object_id]
            assert len(trial_idx) == 1, f'something is off....'
            assert name in tm.array_annotations_at_index(trial_idx).keys(), f'{name} not a valid metaname for trials'
            return tm.array_annotations_at_index(trial_idx)[name][0]

        else:
            if 'clean' in object_id:
                assert name in self.lfp_clean_df.columns, f'{name} not in object annotations'
                return self.lfp_clean_df.loc[object_id, name]
            else:
                return self.lfp_df.loc[object_id, name]

    # -----------------------------------------------------------------------------------
    #                           MISC
    # -----------------------------------------------------------------------------------

    def set_unit_criteria(self, verbose=True, **kwargs):
        if verbose:
            print('Updating unit criteria (removing existing):')
        self.unit_quality_criteria = dict()
        for kwarg in kwargs.items():
            if verbose:
                print(f'\t{kwarg[0]} - {kwarg[1]}')
            self.unit_quality_criteria[kwarg[0]] = kwarg[1]

        self._annotate_good_unit()
        self._get_dataframes(debug=False)

    def _annotate_good_unit(self):
        for sp in self.block.segments[0].spiketrains:
            is_good = True
            excluding_reason = ''
            for uc in self.unit_quality_criteria.items():
                name = uc[0]
                val = uc[1][0]
                opr = uc[1][1]

                if not self._get_operator(opr)(sp.annotations[name], val):
                    is_good = False
                    excluding_reason += '_' + f'{name}'

            # if sp.annotations['is_artefact']:
            #     is_good = False

            sp.annotate(good_unit=is_good)
            sp.annotate(excluding_reason=excluding_reason)

    def _find_noise_trials(self):
        self.noise_trial_ids = []

        ne = self.block.filter(name='noise_epochs')
        if len(ne) == 0:
            return
        ne = ne[0]
        nestart = ne.times.rescale('s').magnitude
        neend = nestart + ne.durations.rescale('s').magnitude
        tids = self.select_trials(all=True, events_of_interest=['sample_start', 'poke'])
        tstart = self.get_event_times(tids, 'sample_start').rescale('s').magnitude
        tend = self.get_event_times(tids, 'poke').rescale('s').magnitude

        for tid, ts, te in zip(tids, tstart, tend):
            ttot = 0  # Total time of noise in trial
            for n0, n1 in zip(nestart, neend):
                t = None
                if n0 < ts and n1 > ts and n1 < te:
                    t = n1 - ts
                elif n0 >= ts and n1 <= te:
                    t = n1 - n0
                elif n0 >= ts and n0 <= te and n1 >= te:
                    t = te - n0
                elif n0 < ts and n1 > te:
                    t = te - ts

                if t is not None:
                    ttot += t

            if ttot is not None:
                if ttot > 0.1:
                    self.noise_trial_ids.append(tid)
                    continue
            dT = te - ts
            if dT > 5:
                self.noise_trial_ids.append(tid)
                continue

    @staticmethod
    def _get_operator(op):
        # Method to select an operator using a string
        operator_dict = {
            '<': operator.lt,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne,
            '>': operator.gt,
            '>=': operator.ge}
        assert op in operator_dict.keys(), '{0} is not a valid operator!'.format(
            op)
        return operator_dict[op]



if __name__ == '__main__':
    io = Io()
    io.load_session(0, read_lfp=True)
    tids = io.select_trials(modality='V')
    # for s in io.block.segments[0].analogsignals:
    #     print(s.name)
    # lfp = io.get_object(io.lfp_clean_ids[0])
    # print(lfp.shape)
    for n in io.unit_metanames:
        print(n)
