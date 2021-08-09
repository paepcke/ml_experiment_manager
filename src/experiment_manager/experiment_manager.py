'''
Created on Aug 5, 2021

@author: paepcke
'''
'''
TODO:
   o Doc that you can store hparams individually as dict,
       or use NeuralNetConfig.
   o Turn into separate project; needs NeuralNetConfig and parts of Utils
   o When saving dataframes with index_col, use that also 
       when using pd.read_csv(fname, index_col) to get the
       index installed
   o Add Series and nparray to data types
   o np.arrays handled
   o checking for row lengths for csv data
   
'''

import csv
import json 
import os
from pathlib import Path
import shutil
import threading

import torch

from experiment_manager.neural_net_config import NeuralNetConfig, ConfigError
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import numpy as np


class ExperimentManager(dict):
    '''
    
TODO:
  o Documentation:
      * Example of mixed saving
      * Deleting an item
    
    Container to hold all information about an experiment:
    the pytorch model parameters, location of model snapshot,
    location of csv files created during training and inference.
    
    An experiment instance is saved and loaded via
        o <exp-instance>.save(), and 
        o ExperimentManager(path)
        
    Storage format is json
    
    Methods:
        o mv                  Move all files to a new root
        o save                Write a pytorch model, csv file, or figure
        
        o add_csv            Create a CSV file writer
        o close_csv           Close a CSV file writer
        
        o close               Close all files
    
    Keys:
        o root_path
        o model_path
        o logits_path
        o probs_path
        o ir_results_path
        o tensor_board_path
        o perf_per_class_path
        o conf_matrix_img_path
        o pr_curve_img_path
    
    '''
    
    #------------------------------------
    # __new__
    #-------------------
    
    def __new__(cls, root_path, initial_info=None):
        '''
        
        :param initial_info: optionally, a dict with already
            known facts about the experiment.
        :type initial_info:
        '''

        self = super().__new__(cls)

        if not os.path.exists(root_path):
            os.makedirs(root_path)
        if not os.path.isdir(root_path):
            raise ValueError(f"Root path arg must be a dir, not {root_path}")

        self.root              = root_path
        self.auto_save_thread  = None
        
        # No csv writers yet; will be a dict
        # of CSV writer instances keyed by file
        # names (without directories):
        self.csv_writers = {}

        if initial_info is not None:
            # Must be a dict:
            if not type(initial_info) == dict:
                raise TypeError(f"Arg initial_info must be a dict, not {initial_info}")

            # Add passed-in info to what we know:
            self.update(initial_info)
            
        # Check whether the given root already contains an
        # 'experiment.json' file:
        experiment_json_path = os.path.join(root_path, 'experiment.json')
        if os.path.exists(experiment_json_path):
            with open(experiment_json_path, 'r') as fd:
                restored_dict_contents = json.load(fd)
                self.update(restored_dict_contents)

        if 'class_names' not in list(self.keys()):
            self['class_names'] = None

        self.models_path       = os.path.join(self.root, 'models')
        self.figs_path         = os.path.join(self.root, 'figs')
        self.csv_files_path    = os.path.join(self.root, 'csv_files')
        self.tensorboard_path  = os.path.join(self.root, 'tensorboard')
        self.hparams_path      = os.path.join(self.root, 'hparams')
        
        self._create_dir_if_not_exists(self.root)
        self._create_dir_if_not_exists(self.models_path)
        self._create_dir_if_not_exists(self.figs_path)
        self._create_dir_if_not_exists(self.csv_files_path)
        self._create_dir_if_not_exists(self.tensorboard_path)
        self._create_dir_if_not_exists(self.hparams_path)
        
        # External info
        self['root_path'] = self.root
        
        # Add internal info so it will be saved
        # by _save_self():
        self['root_path']               = self.root
        self['_models_path']            = self.models_path
        self['_figs_path']                = self.figs_path
        self['_csv_files_path']         = self.csv_files_path
        self['_tensorboard_files_path'] = self.tensorboard_path
        self['_hparams_path']           = self.hparams_path
        
        # Create DictWriters for any already
        # existing csv files:
        self._open_csv_writers()
        
        # Create hparams configurations that might be available:
        self._open_config_files()

        return self

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, _root_path):
        '''
        The __new__() method did most of the work.
        Now, if this is a real new instance, as opposed
        to one that is reconstituted via load(), save
        the experiment for the first time.
        '''

        self._save_self()

    # --------------- Public Methods --------------------

    #------------------------------------
    # save 
    #-------------------
    
    def save(self, item=None, fname=None, index_col=None):
        '''
        Save any of:
            o pytorch model
            o dictionaries
            o lists
            o pd.Series
            o pd.DataFrame
            o figures

            o this experiment itself
            
        If no item is provided, saves this experiment.
        Though it is automatically saved anyway whenever 
        a change is made, and when close() is called.
        
        The fname is the file name with or without .csv extension.
        The file will exist in the self.csv_files_path under the experiment
        root no matter what path is provided by fname. Any parent of 
        fname is discarded. The intended form of fname is just like:
        
            logits
            prediction_numbers
            measurement_results
            
        The index_col is only relevant when saving
        dataframes. If provided, the df index (i.e. the row labels)
        are saved in the csv file with its own column, named
        index_col. Else the index is ignored.
        
        Saving behaviors:
            o Models: 
                 if fname exists, the name is extended
                 with '_<n>' until it is unique among this
                 experiment's already saved models. Uses
                 torch.save
            o Dictionaries and array-likes:
                 If a csv DictWriter for the given fname
                 exists. 
                 
                 If no DictWriter exists, one is created with
                 header row from the dict keys, pd.Series.index, 
                 pd.DataFrame.columns, or for simple lists, range(len())
                 
            o Figures:
                 if fname exists, the name is extended
                 with '_<n>' until it is unique among this
                 experiment's already saved figures. Uses
                 plt.savefig with file format taken from extension.
                 If no extension provided in fname, default is PDF 

        :param item: the data to save
        :type item: {dict | list | pd.Series | pd.DataFrame | torch.nn.Module | plt.Figure}
        :param fname: key for retrieving the file path and DictWriter
        :type fname: str
        :param index_col: for dataframes only: col name for
            the index; if None, index is ignored
        :type index_col: {None | str}
        :return: path to file where given data are stored 
            for persistence
        :rtype: str
        '''
        
        if item is None:
            self._save_self()
            return

        # If item is given, fname must also be provided:
        if fname is None:
            raise ValueError("Must provide file name (i.e. the item key).")

        # Fname is intended for use as the key to the 
        # csv file in self.csv_writers, and as the file
        # name inside of self.csv_files_path. So, clean
        # up what's given to us: remove parent dirs and
        # the extension:
        fname = Path(fname).stem
        
        if type(item) == nn:
            model = item
            # A pytorch model
            dst = os.path.join(self.models_path, fname)
            #if os.path.exists(dst):
            #    dst = self._unique_fname(self.models_path, fname)
            torch.save(model.state_dict(), dst)

        elif type(item) in (dict, list, pd.Series, pd.DataFrame, np.ndarray):
            dst = self._save_records(item, fname, index_col)
            
        elif isinstance(item, NeuralNetConfig):
            self.add_hparams(item, fname)
            dst = os.path.join(self.hparams_path, f"{fname}.json")

        elif type(item) == plt.Figure:
            fig = item
            fname_ext   = Path(fname).suffix
            # Remove the leading period if extension provided:
            file_format = 'pdf' if len(fname_ext) == 0 else fname_ext[1:] 
            dst = os.path.join(self.figs_path, fname)
            #if os.path.exists(dst):
            #    dst = self._unique_fname(self.figs_path, fname)
            plt.savefig(fig, dpi=150, format=file_format)

        self.save()
        return dst

    #------------------------------------
    # add_hparams
    #-------------------
    
    def add_hparams(self, config_fname_or_obj, key):
        '''
        If config_fname_or_obj is a string, it is assumed
        to be a configuration file readable by NeuralNetConfig
        (or the built-in configparser). In that case, 
        read the given file, creating a NeuralNetConfig instance. 
        Store that in self[key]. Also, write a json copy to 
        the hparams subdir, with key as the file name.
        
        If config_fname_or_obj is already a NeuralNetConfig instance,
        write a json copy to the hparams subdir, with key as the file name,
        and store the instance in self[key].
        
        May be called by client, but is also called by save()
        when client calls save() with a config instance.
        
        :param config_fname_or_obj: path to config file that is
            readable by the standard ConfigParser facility. Or
            an already finished NeuralNetConfig instance
        :type config_fname_or_obj: {src | NeuralNetConfig}
        :param key: key under which the config is to be
            available
        :type key: str
        :return a NeuralNetConfig instance 
        :rtype NeuralNetConfig
        '''

        if type(config_fname_or_obj) == str:
            # Was given the path to a configuration file:
            config = self._initialize_config_struct(config_fname_or_obj)
        else:
            config = config_fname_or_obj
            
        self[key] = config
        # Save a json representation in the hparams subdir:
        config_path = os.path.join(self.hparams_path, f"{key}.json")
        config.to_json(config_path, check_file_exists=False)
        return config 

    #------------------------------------
    # update
    #-------------------
    
    def update(self, *args, **kwargs):
        '''
        Save to json every time the dict is changed.
        '''
        super().update(*args, **kwargs)
        self.save()

    #------------------------------------
    # tensorboard_path
    #-------------------
    
    def tensorboard_path(self):
        '''
        Returns path to directory where tensorboard
        files may be held for this experiment.
        '''
        return self.tensorboard_path

    #------------------------------------
    # abspath
    #-------------------
    
    def abspath(self, fname, extension):
        '''
        Given the fname used in a previous save()
        call, and the file extension (.csv or .pth,
        .pdf, etc.): returns the current full path to the
        respective file.
        
        :param fname: name of the item to be retrieved
        :type fname: str
        :param extension: file extension, like 'jpg', 'pdf', 
            '.pth', 'csv', '.csv'
        :type extension: str
        :returns absolute path to corresponding file
        :rtype str
        '''
        if not extension.startswith('.'):
            extension = f".{extension}"
        true_fname = Path(fname).stem + extension
         
        for root, _dirs, files in os.walk(self.root):
            if true_fname in files:
                return os.path.join(root, true_fname)


    #------------------------------------
    # close 
    #-------------------
    
    def close(self):
        '''
        Close all csv writers, and release other resources
        if appropriate
        '''
        
        for csv_writer in self.csv_writers.values():
            # We previously added the fd of the file
            # to which the writer is writing in the 
            # csv.DictWriter instance. Use that now:
            csv_writer.fd.close()
            
        self._save_self()

    #------------------------------------
    # clear 
    #-------------------
    
    def clear(self, safety_str):
        '''
        Removes all results from the experiment.
        Use extreme caution. For safety, the argument
        must be "Yes, do it"
        
        :param safety_str: keyphrase "Yes, do it" to ensure
            caller thought about the call
        :type safety_str: str
        '''
        if safety_str != 'Yes, do it':
            raise ValueError("Saftey passphrase is not 'Yes, do it', so experiment not cleared")
        shutil.rmtree(self.root)


    # --------------- Private Methods --------------------
    
    #------------------------------------
    # _open_config_files 
    #-------------------
    
    def _open_config_files(self, instance=None):
        '''
        Finds files in the hparams subdirectory. Any
        files there are assumed to be json files from
        previously saved NeuralNetConfig instances.
        
        Recreates those instances, and sets value of
        the corresponding keys to those instances. Keys
        are the file names without extension.
          
        :param instance: if provided, the instance whose keys
            are to be set instead of self.
        :type instance: ExperimentManager
        '''
        
        # If the hparams path contains json files of
        # saved configs, turn them into NeuralNetConfig instances,
        # and assign those to self[<keys>] with one key
        # for each hparam json file (usually that will just
        # be one):
        
        if instance is not None:
            self = instance
        
        for file in os.listdir(self.hparams_path):
            path = os.path.join(self.hparams_path, file)
            with open(path, 'r') as fd:
                config_json_str = fd.read()
            config = NeuralNetConfig.from_json(config_json_str)
            key = Path(path).stem
            self[key] = config

    #------------------------------------
    # _open_csv_writers
    #-------------------
    
    def _open_csv_writers(self, instance=None):
        '''
        Finds csv files in the csv subdirectory,
        opens a DictWriter for each, and adds the
        writer under the file name key.
        
        :param instance: if provided, the initialization
            of key/val pairs will occur on that instance,
            instead of self. Used only when called from
            __new__()
        :type instance: ExperimentManager
        '''
        
        if instance is not None:
            self = instance
        for file in os.listdir(self.csv_files_path):
            path = os.path.join(self.csv_files_path, file)
            # Sanity check:
            if Path(path).suffix == '.csv':
                # Get the field names (i.e. header row):
                with open(path, 'r') as fd:
                    col_names = csv.DictReader(fd).fieldnames
            
                # Make the writer:
                with open(path, 'a') as fd:
                    writer = csv.DictWriter(fd, col_names)
                    writer.fd = fd
                    key = Path(path).stem
                    self[key] = writer.fd.name
                    self.csv_writers[key] = writer 

    #------------------------------------
    # _schedule_save 
    #-------------------
    
    def _schedule_save(self):
        '''
        If no self-save task is scheduled yet,
        schedule one:
        '''
        try:
            # Only schedule a save if none
            # is scheduled yet:
            if self.auto_save_thread is not None and \
                not self.auto_save_thread.cancelled():
                return
            self.auto_save_thread = AutoSaveThread(self.save)
            self.auto_save_thread.start()
        except Exception as e:
            raise ValueError(f"Could not schedule an experiment save: {repr(e)}")

    #------------------------------------
    # _cancel_save 
    #-------------------
    
    def _cancel_save(self):
        '''
        Cancel all self-save tasks:
        '''
        try:
            if self.auto_save_thread is not None:
                self.auto_save_thread.cancel()
        except Exception as e:
            raise ValueError(f"Could not cancel an experiment save: {repr(e)}")

    #------------------------------------
    #_save_records 
    #-------------------

    def _save_records(self, item, fname, index_col=None, trust_list_dim=True):
        '''
        Saves items of types dict, list, Pandas Series,
        numpy arrays, and DataFrames to a csv file. Creates the csv
        file and associated csv.DictWriter if needed. 
        If DictWriter has to be created, adds it to the
        self.csv_writers dict under the fname key.
        
        When creating DictWriters, the header line (i.e. column
        names) is obtain from:
        
            o keys() if item is a dict,
            o index if item is a pd.Series
            o columns if item is a pd.DataFrame
            o range(top-level-num-columns) if 
                item is a Python list or numpy array
                
        It is a ValueError for item to be an array-like with
        3 or more dimensions.
        
        If DictWriter already exists, adds the record(s)

        The fname is used as a key into self.csv_writers, and
        is expected to not be a full path, or to have an extension
        such as '.csv'. Caller is responsible for the cleaning.
        
        The index_col is relevant only for dataframes: if None,
        the df's index (i.e. the row labels) are ignored. Else,
        the index values are stored as a column with column name
        index_col.
        
        The trust_list_dim is relevant only for 2D lists. If True,
        trust that all rows of the list are the same length. Else
        each row's length is checked, and a ValueError thrown if
        lengths are unequal. 
            
        :param item: data to be written to csv file
        :type item: {dict | list | pd.Series | pd.DataFrame}
        :param fname: name for the csv file stem, and retrieval key
        :type fname: str
        :param index_col: for dataframes only: name of index
            column. If None, index will be ignored
        :type index_col: {str | None}
        :param trust_list_dim: for 2D lists only: trust that all
            rows are of equal lengths
        :type trust_list_dim: True
        :return full path to the csv file
        :rtype str
        '''

        # Do we already have a csv writer for the given fname?
        dst = os.path.join(self.csv_files_path, f"{fname}.csv")
        #if os.path.exists(dst):
        #    dst = self._unique_fname(self.csv_files_path, fname)

        # Do we already have csv writer for this file:
        try:
            csv_writer = self.csv_writers[fname]
        except KeyError:
            # No CSV writer yet:
            header = self._get_field_names(item, index_col=index_col, trust_list_dim=trust_list_dim)

            fd = open(dst, 'w')
            csv_writer = csv.DictWriter(fd, header)
            # Save the fd with the writer obj so
            # we can flush() when writing to it:
            csv_writer.fd = fd
            csv_writer.writeheader()
            self.csv_writers[fname] = csv_writer
        else:
            header = csv_writer.fieldnames
            # If item is df, we need to determine
            # whether the index should be included
            # in a column:
            if type(item) == pd.DataFrame:
                col_names = item.columns
                # If we have one more fld name than
                # the number of cols, then assume that
                # the first fld name is for the index column:
                if len(header) == len(col_names) + 1:
                    index_col = header[0]
                else:
                    index_col = None

        # Now the DictWriter exists; write the data.
        # Method for writing may vary with data type.

        # For pd.Series, use its values as a row;
        # for lists, 
        if type(item) == pd.Series:
            item = list(item)
        elif type(item) == list:
            item = np.array(item)

        # If given a dataframe, write each row:
        if type(item) == pd.DataFrame:
            num_dims = len(item.shape)
            if num_dims > 2:
                raise ValueError(f"For dataframes, can only handle 1D or 2D, not {item}")
            
            if index_col is None:
                # Get to ignore the index (i.e. the row labels):
                for row_dict in item.to_dict(orient='records'):
                    csv_writer.writerow(row_dict)
            else:
                for row_dict in self._collapse_df_index_dict(item, index_col):
                    csv_writer.writerow(row_dict)
                    
        # Numpy array or Python list:
        elif type(item) in(np.ndarray, list):
            num_dims = len(self._list_shape(item)) if type(item) == list else len(item.shape)
            if num_dims == 1:
                csv_writer.writerow(self._arr_to_dict(item, header))
            else:
                for row in item:
                    csv_writer.writerow(self._arr_to_dict(row, header))

        # A dict:
        else:
            # This is a DictWriter's native food:
            csv_writer.writerow(item)
            
        csv_writer.fd.flush()
        return dst


    #------------------------------------
    # _get_field_names
    #-------------------
    
    def _get_field_names(self, item, index_col=None, trust_list_dim=True):
        '''
        Given a data structure, return the column header
        fields appropriate to the data

        Raises ValueError if the dimension of data is not 1D or 2D.
        The trust_list_dim is relevant only if item is a Python list.
        The arg controls whether the number of columns in the list
        is constant across all rows. If trust_list_dim is False,
        the length of each row is checked, which forces a loop 
        through the list. Even with trust_list_dim is False, the 
        dimensions of the list are checked to be 1D or 2D.

        Strategy for determining a column header, given type of item:
           o dict: list of keys
           o np.ndarray or Python list: range(num-columns)
           o pd.Series: index
           o pd.DataFrame: columns
        
        :param item: data structure from which to deduce
            a header
        :type item: {list | np.ndarray | pd.Dataframe | pd.Series | dict}
        :param index_col: only relevant if item is a dataframe.
            In that case: column name to use for the index column.
            If None, index will not be included in the columns.
        :type index_col: {None | str}
        :returns the header
        :rtype [str]
        :raises ValueError if dimensions are other than 1, or 2
        '''
        
        bad_shape = False
        
        # Get dimensions of list or numpy array
        if type(item) == list:
            dims = self._list_shape(item)
        elif type(item) == np.ndarray:
            dims = item.shape

        if type(item) == np.ndarray or type(item) == list:
            if len(dims) == 1:
                header = list(range(dims[0]))
            elif len(dims) == 2:
                header = list(range(dims[1]))
            else:
                bad_shape = True
            # When no index given to Series, col names will
            # be integers (0..<len of series values>).
            # Turn them into strs as expected by callers:
            if type(header[0]) == int:
                header = [str(col_name) for col_name in header]

        elif type(item) == dict:
            header = list(item.keys())
        elif type(item) == pd.Series:
            header = item.index.to_list()
            # When no index given to Series, col names will
            # be integers (0..<len of series values>).
            # Turn them into strs as expected by callers:
            if type(header[0]) == int:
                header = [str(col_name) for col_name in header]
        elif type(item) == pd.DataFrame:
            header = item.columns.to_list()
            # Add a column name for the row labels:
            if index_col is not None:
                header = [index_col] + header
        else:
            raise TypeError(f"Can only store dicts and list-like, not {item}")
        
        # Item is not 1 or 2D:
        if bad_shape:
            raise ValueError(f"Can only handle 1D or 2D, not {item}")
        
        # Is item a likst, and we were asked to 
        # check each row? 
        if type(item) == list and len(dims) == 2 and not trust_list_dim:
            # We know by now that list is 2D, check that
            # all rows are the same length
            len_1st_row = len(item[0])
            for row_num, row in enumerate(item):
                if len(row) != len_1st_row:
                    raise ValueError(f"Inconsistent list row length in row {row_num}")
        return header

    #------------------------------------
    # _list_shape
    #-------------------
    
    def _list_shape(self, list_item):
        '''
        
        :param list_item:
        :type list_item:
        '''
        if not type(list_item) == list:
            return []
        return [len(list_item)] + self._list_shape(list_item[0])

    #------------------------------------
    # _arr_to_dict 
    #-------------------
    
    def _arr_to_dict(self, arr1D, fieldnames):
        '''
        Return a dict constructed from a 1D array.
        Key names are taken from the given csv.DictWriter's
        fieldnames property. arr1D may be a 1D Python list,
        or a pandas Series.
        
        :param arr1D: array to convert
        :type arr1D: [any]
        :param fieldnames: list of column names
        :type [str]
        :return dictionary with keys being the fieldnames
        '''
        if len(arr1D) != len(fieldnames):
            raise ValueError(f"Inconsistent shape of arr ({arr1D}) for fieldnames ({fieldnames})")
        tmp_dict = {k : v for k,v in zip(fieldnames, arr1D)}
        return tmp_dict

    #------------------------------------
    # _collapse_df_index_dict
    #-------------------

    def _collapse_df_index_dict(self, df, index_col):
        '''
        Given a df, return a dict that includes the
        row indices (i.e. row labels) in the column names
        index_col. Example: given dataframe:

                  foo  bar  fum
            row1    1    2    3
            row2    4    5    6
            row3    7    8    9
        
        and index_col 'row_label', return:

            [
              {'row_label' : 'row1': 'foo': 1, 'bar': 2, 'fum': 3}, 
              {'row_label' : 'row2', 'foo': 4, 'bar': 5, 'fum': 6}, 
              {'row_label' : 'row3': 'foo': 7, 'bar': 8, 'fum': 9}
            ]

        :param df: dataframe to collapse
        :type df: pd.DataFrame
        :return array of dicts, each corresponding to one
            dataframe row
        :rtype [{str: any}]
        '''
        # Now have
        df_nested_dict = df.to_dict(orient='index')
        # Now have:
        #  {'row1': {'foo': 1, 'bar': 2, 'fum': 3}, 'row2': {'foo': 4, ...
        df_dicts = []
        for row_label, row_rest_dict in df_nested_dict.items():
            df_dict = {index_col : row_label}
            df_dict.update(row_rest_dict)
            df_dicts.append(df_dict)
        return df_dicts

    #------------------------------------
    # _initialize_config_struct 
    #-------------------
    
    def _initialize_config_struct(self, config_info):
        '''
        Return a NeuralNetConfig instance, given
        either a configuration file name, or a JSON
        serialization of a configuration.

          config['Paths']       -> dict[attr : val]
          config['Training']    -> dict[attr : val]
          config['Parallelism'] -> dict[attr : val]
        
        The config read method will handle config_info
        being None. 
        
        If config_info is a string, it is assumed either 
        to be a file containing the configuration, or
        a JSON string that defines the config.
        
        :param config_info: the information needed to construct
            the NeuralNetConfig instance: file name or JSON string
        :type config_info: str
        :return a NeuralNetConfig instance with all parms
            initialized
        :rtype NeuralNetConfig
        '''

        if isinstance(config_info, str):
            # Is it a JSON str? Should have a better test!
            if config_info.startswith('{'):
                # JSON String:
                config = NeuralNetConfig.from_json(config_info)
            else: 
                config = NeuralNetConfig(config_info)
        else:
            msg = f"Error: must pass a config file name or json, not {config_info}"
            raise ConfigError(msg)
            
        return config


    #------------------------------------
    # _save_self 
    #-------------------
    
    def _save_self(self):
        '''
        Write json of info about this experiment
        to self.root/experiment.json
        '''
        
        # Insist on the class names to have been set:
        #try:
        #    self['class_names']
        #except KeyError:
        #    raise ValueError("Cannot save experiment without class_names having been set first")

        # If config facility is being used, turn
        # the NeuralNetConfig instance to json:
        try:
            config = self['config']
            if isinstance(config, NeuralNetConfig):
                self['config'] = config.to_json()
        except:
            # No config
            pass
        
        with open(os.path.join(self.root, 'experiment.json'), 'w') as fd:
            json.dump(self, fd)
            
        self._cancel_save()

    #------------------------------------
    # _is_experiment_file 
    #-------------------
    
    def _is_experiment_file(self, path):
        '''
        Return True if the given path is 
        below the experiment root directory
        
        :param path: absolute path to check
        :type path: str
        :return whether or not path is below root
        :rtype bool
        '''
        if type(path) == str and path.startswith(self.root):
            return True
        else:
            return False


    #------------------------------------
    # _create_dir_if_not_exists 
    #-------------------
    
    def _create_dir_if_not_exists(self, path):
        
        if not os.path.exists(path):
            os.makedirs(path)
            return
        # Make sure the existing path is a dir:
        if not os.path.isdir(path):
            raise ValueError(f"Path should be a directory, not {path}")

    #------------------------------------
    # _path_elements 
    #-------------------
    
    def _path_elements(self, path):
        '''
        Given a path, return a dict of its elements:
        root, fname, and suffix. The method is almost
        like Path.parts or equivalent os.path method.
        But the 'root' may be absolute, or relative.
        And fname is provided without extension.
        
          foo/bar/blue.txt ==> {'root' : 'foo/bar',
                                'fname': 'blue',
                                'suffix: '.txt'
                                }

          /foo/bar/blue.txt ==> {'root' : '/foo/bar',
                                'fname': 'blue',
                                'suffix: '.txt'
                                }

          blue.txt ==> {'root' : '',
                        'fname': 'blue',
                        'suffix: '.txt'
                        }
        
        :param path: path to dissect
        :type path: str
        :return: dict with file elements
        :rtype: {str : str}
        '''
        
        p = Path(path)
        
        f_els = {}
        
        # Separate the dir from the fname:
        # From foo/bar/blue.txt  get ('foo', 'bar', 'blue.txt')
        # From /foo/bar/blue.txt get ('/', 'foo', 'bar', 'blue.txt')
        # From blue.txt          get ('blue.txt',)
        
        elements = p.parts
        if len(elements) == 1:
            # just blue.txt
            f_els['root']   = ''
            nm              = elements[0]
            f_els['fname']  = Path(nm).stem
            f_els['suffix'] = Path(nm).suffix
        else:
            # 
            f_els['root']     = os.path.join(*list(elements[:-1]))
            f_els['fname']    = p.stem
            f_els['suffix']   = p.suffix
        
        return f_els


    #------------------------------------
    # _unique_fname 
    #-------------------
    
    def _unique_fname(self, out_dir, fname):
        '''
        Returns a file name unique in the
        given directory. I.e. NOT globally unique.
        Keeps adding '_<i>' to end of file name.

        :param out_dir: directory for which fname is to 
            be uniquified
        :type out_dir: str
        :param fname: unique fname without leading path
        :type fname: str
        :return: either new, or given file name such that
            the returned name is unique within out_dir
        :rtype: str
        '''
        
        full_path   = os.path.join(out_dir, fname)
        fname_dict  = self._path_elements(full_path)
        i = 1

        while True:
            try:
                new_path = os.path.join(fname_dict['root'], fname_dict['fname']+fname_dict['suffix'])
                with open(new_path, 'r') as _fd:
                    # Succeed in opening, so file exists.
                    fname_dict['fname'] += f'_{i}'
                    i += 1
            except:
                # Couldn't open, so doesn't exist:
                return new_path

    #------------------------------------
    # __setitem__
    #-------------------
    
    def __setitem__(self, key, item):
        '''
        Save to json every time the dict is changed.
        
        :param key: key to set
        :type key: str
        :param item: value to map to
        :type item: any
        '''
        super().__setitem__(key, item)
        self._schedule_save()

    #------------------------------------
    # __delitem__
    #-------------------
    
    def __delitem__(self, key):
        
        # Allow KeyError to bubble up to client,
        # if the key doesn't exist:
        item = self[key]
        
        # If a DictWriter, close it, and delete the file:
        if type(item) == csv.DictWriter:
            path = item.fd.name
            # Only delete file if it's under the
            # exeriment root. Else it could be a
            # DictWriter saved under a top level
            # user key, and owned by them:
            if self._is_experiment_file(path):
                item.fd.close()
                os.remove(path)
        elif self._is_experiment_file(item):
            # If this is a file in the experiment
            # tree, delete it:
            os.remove(item)

        # Delete the key/val pair:
        super().__delitem__(key)
        self._schedule_save()


# ------------------- Class AutoSaveThread -----------------

class AutoSaveThread(threading.Thread):
    '''
    Used to save an experiment after a delay. Operations
    on AutoSaveThread instances:
    
        o start()
        o cancel()
        o cancelled()
        
    The class can actually be used with any callable.
    Functionality is like the built-in sched, but
    the action is one-shot. After the function has
    been called, the thread terminates.
    
    Usage examples:
            AutoSaveThread(experiment.save).start()
            AutoSaveThread(experiment.save, time_delay=5).start()
            
    '''
    
    DEFAULT_DELAY = 2 # seconds
    
    # Condition shared by all AutoSaveThread threads:
    _CANCEL_CONDITION = threading.Condition()

    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, call_target, *args, time_delay=None, **kwargs):
        '''
        Setup the action. The call_target can be
        any callable. It will be called with *args
        and **kwargs.
         
        :param call_target: a callable that will be 
            invoked with *args and **kwargs
        :type call_target: callable
        :param time_delay: number of seconds to wait
            before action
        :type time_delay: int
        '''
        super().__init__()
        if time_delay is None:
            self.time_delay = self.DEFAULT_DELAY
        else:
            self.time_delay = time_delay
            
        self.call_target = call_target
        self.args   = args
        self.kwargs = kwargs
        
        self._canceled = threading.Event()
        
    #------------------------------------
    # run 
    #-------------------
    
    def run(self):
        self._CANCEL_CONDITION.acquire()
        self._CANCEL_CONDITION.wait_for(self.cancelled, timeout=self.time_delay)
        self._CANCEL_CONDITION.release()
        if not self.cancelled():
            self.call_target(*self.args, **self.kwargs)

    #------------------------------------
    # cancel
    #-------------------
    
    def cancel(self):
        self._canceled.set()
        try:
            self._CANCEL_CONDITION.notify_all()
        except RuntimeError:
            # Nobody was waiting
            pass

    #------------------------------------
    # cancelled 
    #-------------------
    
    def cancelled(self):
        return self._canceled.is_set()
    
    #------------------------------------
    # delay 
    #-------------------
    
    def delay(self):
        '''
        Returns the delay set for the
        action
        '''
        return self.time_delay
        

