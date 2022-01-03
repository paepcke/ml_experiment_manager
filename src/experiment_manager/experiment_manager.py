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
from enum import Enum
import json 
import os
from pathlib import Path
import re
import shutil
import threading

from PIL import UnidentifiedImageError
import skorch
import torch

from experiment_manager.neural_net_config import NeuralNetConfig, ConfigError
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn


# Values are just for (some) clarity,
# but they must be different from each other:
class Datatype(Enum):
    tabular 	= '.csv'
    model   	= ['.pth', '.pkl']
    figure  	= '.pdf'
    hparams     = 'hparams_json'
    tensorboard = 'tensorboard'
    txt         = '.txt' 
    untyped     = ''

class JsonDumpableMixin:
    '''
    Mixin for derived classes that promise to implement
    method json_dump(file_name). That method must
    write proper json to the file.
    
    The derived classes must also implement json.load(fname),
    which must return an instance of the derived class. 
    '''
    def json_dump(self, fname):
        raise NotImplementedError("Subclass must implement this method")
    def json_load(self, fname):
        raise NotImplementedError("Subclass must implement this method")

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
        o read                Recover a previously saved item
        o destroy             Remove a previously saved item
        o abspath             Path to where an item is stored
        o listdir             List all saved files of a given type
        
        o add_hparams
        o add_csv             Create a CSV file writer
        o close_csv           Close a CSV file writer
        o col_names           Column names of tabular data (dataframes, series, dicts, etc.)
        o tensorboard_path    Path to where the tensorboard info is written
        
        o clear               Remove all saved items
        o close               Close all files

    
    Keys:
        o root
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
    # Constructor 
    #-------------------

    def __init__(self, root_path, initial_info=None):
        '''
        If the given directory already contains an experiment.json
        file, the resulting instance will be contain all
        the information of the already existing experiment.
        Else a new experiment is created.
        
        The initial_info is an optional dict that will be
        used to set desired initial key-value pairs in the
        old or new instance.
        
        :param root_path: path to root directory of the
            experiment to be created or loaded
        :type root_path: str
        :param initial_info: optionally, a dict with already
            known facts about the experiment.
        :type initial_info: {str : any}
        '''

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

        self.models_path        = os.path.join(self.root, 'models')
        self.figs_path          = os.path.join(self.root, 'figs')
        self.csv_files_path     = os.path.join(self.root, 'csv_files')
        self.tensorboard_path   = os.path.join(self.root, 'tensorboard')
        self.hparams_path       = os.path.join(self.root, 'hparams')
        self.json_files_path    = os.path.join(self.root, 'json_files')
        self.txt_files_path     = os.path.join(self.root, 'txt_files')
        self.untyped_files_path = os.path.join(self.root, 'untyped_files')

        self._create_dir_if_not_exists(self.root)
        self._create_dir_if_not_exists(self.models_path)
        self._create_dir_if_not_exists(self.figs_path)
        self._create_dir_if_not_exists(self.csv_files_path)
        self._create_dir_if_not_exists(self.tensorboard_path)
        self._create_dir_if_not_exists(self.hparams_path)
        self._create_dir_if_not_exists(self.json_files_path)
        self._create_dir_if_not_exists(self.txt_files_path)        
        self._create_dir_if_not_exists(self.untyped_files_path)

        # Where to put which kind of Datatype or item Python type:
        self.dir_dict = {
            Datatype.tabular : self.csv_files_path,
            Datatype.model   : self.models_path,
            Datatype.figure  : self.figs_path,
            Datatype.hparams : self.hparams_path,
            Datatype.tensorboard : self.tensorboard_path,
            dict             : self.csv_files_path,
            list             : self.csv_files_path,
            pd.Series        : self.csv_files_path,
            pd.DataFrame     : self.csv_files_path,
            np.ndarray       : self.csv_files_path,
            JsonDumpableMixin: self.json_files_path,
            Datatype.txt     : self.txt_files_path,
            Datatype.untyped : self.untyped_files_path
        }
        
        
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
        self['_json_files_path']        = self.json_files_path
        self['_txt_files_path']         = self.txt_files_path
        self['_untyped_files_path']     = self.untyped_files_path

        # Create DictWriters for any already
        # existing csv files:
        self._open_csv_writers()
        
        # Create hparams configurations that might be available:
        self._open_config_files()

        self._save_self()

    # --------------- Public Methods --------------------

    #------------------------------------
    # save 
    #-------------------
    
    def save(self, key=None, item=None, index_col=None, header=None):
        '''
        Save any of:
            o pytorch model
            o dictionaries
            o lists
            o pd.Series
            o pd.DataFrame
            o figures
            o NeuralNetConfig instances
            o instances of classes that inherit from JsonDumpableMixin
            o strings are assumed to be tensorboards

            o this experiment itself
            
        If neither key nor item is provided, saves this experiment.
        Though all state is automatically saved anyway whenever 
        a change is made, and when close() is called.
        
        The key is used as a file name. The file will be created
        under the experiment root with an extension appropriate to
        the information type. The intended form of key is like:
        
            'logits'
            'prediction_numbers'
            'measurement_results'
            
        The index_col is only relevant when saving
        dataframes. If provided, the df index (i.e. the row labels)
        are saved in the csv file with its own column, named
        index_col. Else the index is ignored.
        
        The header argument may be used for tabular data ahead of
        saving any data. Useful if data will be saved as Python
        lists or np arrays. In those cases the data themselves do
        not reveal a header, as dicts, dataframes, and series instances 
        do. If one of these self-revealing data are saved in a first
        call to this method with the given key, then no prior call
        providing a header is required.
        
        A call to this method with just a key and header will start
        a csv file, writing the header row. It is an error to provide
        a header after a call to this method that saved data. It is 
        an error to provide both header and item.  
        
        Saving behaviors:
            o Models: 
                 if key exists, the name is extended
                 with '_<n>' until it is unique among this
                 experiment's already saved models. Uses
                 torch.save
            o Dictionaries and array-likes:
                 If a csv DictWriter for the given key
                 exists. 
                 
                 If no DictWriter exists, one is created with
                 header row from the dict keys, pd.Series.index, 
                 pd.DataFrame.columns, or for simple lists, range(len())
                 
            o Figures:
                 if key exists, the name is extended
                 with '_<n>' until it is unique among this
                 experiment's already saved figures. Uses
                 plt.savefig with file format taken from extension.
                 If no extension provided in key, default is PDF

            o JsonDumpableMixin instances:
                 the extension '.json' is added to the key if needed.
                 Then method json_dump() is called on item with the
                 (possibly modified) key.  
                 
            o Tensorboard:
                 if key is a string, it is assumed to pertain to 
                 tensorboard management. A subdirectory of name <key>
                 is created below the exp root. Argument <item> is
                 ignored 

        :param key: key for retrieving the file path and DictWriter
        :type key: str
        :param item: the data to save
        :type item: {dict | list | pd.Series | pd.DataFrame | torch.nn.Module | plt.Figure}
        :param index_col: for dataframes only: col name for
            the index; if None, index is ignored
        :type index_col: {None | str}
        :param header: use given list as a header row; used only
            without also providing item, and before a first save
            of data with the given key
        :type header: [str]
        :return: path to file where given data are stored 
            for persistence
        :rtype: str
        :raise ValueError for inconsistent values in arguments
        :raise TypeError for items with unrecognized type
        '''
        if item is None and key is None:
            # Save this experiment itself.
            # Happens periodically after any
            # changes; so client needs not worry
            self._save_self()
            return

        # If item is given, key must also be provided:
        if key is None:
            raise ValueError("Must provide an item key).")
        
        if header is not None:
            # Item must be None, and the key must not
            # yet exist, i.e. not allowed to add a header
            # after a csv file is already started:
            if item is not None:
                raise ValueError("If supplying a header, item argument must be None")
            if key in self.keys():
                fpath = self.abspath(key, Datatype.tabular)
                raise ValueError(f"A header can only be provided if csv file does not exist yet ({fpath})")

        # Key is is used as the key to the 
        # csv file in self.csv_writers, and as the file
        # name inside of self.csv_files_path. So, clean
        # up what's given to us: remove parent dirs and
        # the extension:
        try:
            key = Path(key).stem
        except Exception as _e:
            raise ValueError(f"First argument must be a data access key, not {key}")

        if isinstance(item, skorch.classifier.NeuralNet):
            model = item
            dst = os.path.join(self.models_path, f"{key}.pkl")
            optimizer_path = os.path.join(self.models_path, 'opt.pkl')
            history_path   = os.path.join(self.models_path, 'history.json')
            model.save_params(dst, f_optimizer=optimizer_path, f_history=history_path)

        if isinstance(item, nn.Module):
            model = item
            # A pytorch model
            dst = os.path.join(self.models_path, f"{key}.pth")
            #if os.path.exists(dst):
            #    dst = self._unique_fname(self.models_path, key)
            torch.save(model.state_dict(), dst)

        elif header is not None or type(item) in (dict, list, pd.Series, pd.DataFrame, np.ndarray):
            # Anything tabular, or None, if just writing a header
            # to start a CSV file:
            dst = self._save_records(item, key, index_col=index_col, header=header)
            
        elif isinstance(item, NeuralNetConfig):
            self.add_hparams(key, item)
            dst = os.path.join(self.hparams_path, f"{key}.json")

        elif type(item) == plt.Figure:
            fig = item
            fname_ext   = Path(key).suffix
            # Remove the leading period if extension provided:
            file_format = 'pdf' if len(fname_ext) == 0 else fname_ext[1:] 
            dst = os.path.join(self.figs_path, f"{key}.{file_format}")
            
            fig.savefig(dst, dpi=150, format=file_format)

        elif isinstance(item, JsonDumpableMixin):
            if Path(key).suffix != '.json':
                fname = key + '.json'
            else:
                fname = key
            dst =  os.path.join(self.json_files_path, fname)
            item.json_dump(dst)
            
        elif type(item) == str:
            if Path(key).suffix != '.txt':
                fname = key + '.txt'
            else:
                fname = key
            dst = os.path.join(self.txt_files_path, fname)
            with open(dst, 'w') as fd:
                fd.write(item)

        elif type(key) == str and item is None:
            # Assume this starts a tensorboard data repo.
            dst = os.path.join(self.tensorboard_path, key)
            # If not given a tensorboard SummaryReader, create one:
            os.makedirs(dst, exist_ok=True)

        elif key is not None:
            if Path(key).suffix != '.txt':
                fname = key + '.txt'
            else:
                fname = key
            dst = os.path.join(self.untyped_files_path, fname)
            with open(dst, 'w') as fd:
                fd.write(str(item))

        else:
            raise TypeError(f"Don't know how to save item of type {type(item)}")
        
        # Update the saved state of this experiment instance
        self.save()
        return dst

    #------------------------------------
    # destroy
    #-------------------
    
    def destroy(self, key, datatype):
        '''
        Inverse of save(): close any csv reader on the
        experiment-controlled file to which key refers.
        Then delete the file.
        
        :param key:
        :type key:
        '''
        if type(datatype) != Datatype and not issubclass(datatype, JsonDumpableMixin):
            raise TypeError(f"Data type argument must be a Datatype enum member, not {datatype}")
        
        path = self.abspath(key, datatype)
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find file/dir corresponding to key {key} of type Datatype.{datatype.name}")

        if datatype == Datatype.tabular:
            try:
                # If there is an open writer, close it:
                writer = self.csv_writers[key]
                writer.fd.close()
            except KeyError:
                # No writer open; OK
                pass
            os.remove(path)

        elif datatype in (Datatype.model, Datatype.figure, 
                          Datatype.hparams, Datatype.txt): 
            os.remove(path)
            
        elif datatype == Datatype.tensorboard:
            shutil.rmtree(self.tensorboard_path, ignore_errors=True)
            os.makedirs(self.tensorboard_path)

        elif type(datatype) == type and issubclass(datatype, JsonDumpableMixin):
            os.remove(path)

    #------------------------------------
    # add_hparams
    #-------------------
    
    def add_hparams(self, key, config_fname_or_obj):
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
        
        :param key: key under which the config is to be
            available
        :type key: str
        :param config_fname_or_obj: path to config file that is
            readable by the standard ConfigParser facility. Or
            an already finished NeuralNetConfig instance
        :type config_fname_or_obj: {src | NeuralNetConfig}
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
    # tensorboard_path
    #-------------------
    
    def tensorboard_path(self):
        '''
        Returns path to directory where tensorboard
        files may be held for this experiment.
        '''
        return self.tensorboard_path

    #------------------------------------
    # read
    #-----
    
    def read(self, key, datatype, uninitialized_net=None):
        '''
        Given the key used in a previous save()
        call, and the datatype (Datatype.tabular, 
        Datatype.model, etc.): returns the current 
        respective data in appropriate form. For
        the Datatype enum being:
        
           tabular        Pandas DataFrame
           model          torch.nn
           figure         pyplot Figure
           hparams        NeuralNetConfig
           tensorboard    Path to tensorboard information
           json           instance of class with mixin JsonDumpableMixin
           str            pure string, such as a README
           
        If the datatype is a class with a
        JsonDumpableMixin mixin, then that class' json_load() 
        classmethod is called with the file path, and the resulting 
        instance is returned. 

        For reading skorch models requires the initialized_skorch_net
        kwarg. The saved information will be added to
        that initialized net.
           
        Note: if for tabular data the client rather works with 
            a csv reader for row-by-row processing the following
            or similar could be used:
            
            path = csv.abspath(<key>, Datatype.tabular)
            with open(path, 'r') as fd:
                reader = csv.DictReader(fd)
                for row_dict in reader:
                    ...
        
        :param key: name of the item to be retrieved
        :type key: str
        :param datatype: whether the key refers to 
            a table (i.e. csv file), a figure (.pdf/.png, etc),
            or one of the other Datatype enums
        :type datatype: Datatype
        :param uninitialized_net: initialized neural net; only
            needed for models 
        :type uninitialized_net: {skorch.classifier.NeuralNet | torch.nn.Module
        :returns retrieved item
        :rtype {any}
        :raise FileNotFoundError if item not found
        '''
        
        if type(datatype) != Datatype and not issubclass(datatype, JsonDumpableMixin):
            raise TypeError(f"Data type argument must be a Datatype enum member, not {datatype}")
        
        path = self.abspath(key, datatype)
        not_exists_err_msg = f"Cannot find file/dir corresponding to key '{key}'" 
        if path is None or not os.path.exists(path):
            raise FileNotFoundError(not_exists_err_msg)
        
        # For Datatype.model, the suffix will be
        # .pth for pytorch, or .pkl for skorch:
        if datatype == Datatype.model:
            root = Path(path).parent
            pytorch_model_path = root.joinpath(Path(path).stem + '.pth')
            skorch_model_path  = root.joinpath(Path(path).stem + '.pkl')
            if not pytorch_model_path.exists() and not skorch_model_path.exists():
                raise FileNotFoundError(not_exists_err_msg)
        else:
            if path is None or not os.path.exists(path):
                raise FileNotFoundError(not_exists_err_msg)

        if datatype == Datatype.tabular:
            return pd.read_csv(path)
        
        elif datatype == Datatype.model:
            # Could be a pytorch model (.pth) or
            # a skorch model (.pkl) (though underneath
            # they are all pytorch state_dict exports.
            # But for skorch we also have optimizer and
            # history state, so loading is different:
            return self._load_model(path, uninitialized_net)

        elif datatype == Datatype.figure:
            try:
                return plt.imread(path)
            except UnidentifiedImageError:
                raise TypeError(f"File exists, but cannot read file of type {Path(path).suffix} as an image")
        elif datatype == Datatype.hparams:
            with open(path, 'r') as fd:
                if Path(path).suffix == '.json':
                    json_str = fd.read()
                    return NeuralNetConfig.json_loads(json_str)
                else:
                    # Assume it's a cfg file, and hope  for the best:
                    return NeuralNetConfig(path)
        elif datatype == Datatype.tensorboard:
            return path
        
        elif datatype == Datatype.txt:
            with open(path, 'r') as fd:
                txt = fd.read()
                return txt

        elif issubclass(datatype, JsonDumpableMixin): 
            new_inst = datatype.json_load(path)
            return new_inst

    #------------------------------------
    # col_names
    #-------------------
    
    def col_names(self, key):
        '''
        Retrieve the column name of tabularly stored
        data. I.e. dataframes, series, dicts, lists, 
        numpy arrays.
        
        :param key: data item's key
        :type key: str
        :return the field (i.e. column) names in the 
            csv header
        :rtype [str]
        :raises KeyError if key does not exist for any
            tabular data
        '''
        
        try:
            writer = self.csv_writers[key]
        except KeyError:
            raise KeyError(f"Experiment stores no tabular data under key '{key}'")
        return writer.fieldnames


    #------------------------------------
    # abspath
    #-------------------
    
    def abspath(self, key, datatype):
        '''
        Given the key used in a previous save()
        call, and the datatype (Datatype.tabular, 
        Datatype.model, etc.): returns the path
        to the file where the item is stored. If
        a file of that key does not exist, returns
        None.
        
        :param key: name of the item whose filename
            is to be retrieved
        :type key: str
        :param datatype: the Datatype of the item
        :type extension: Datatype enum member
        :returns absolute path to corresponding file, or
            None if file does not exist
        :rtype str
        :raise TypeError for incorrect argument type
        :
        '''
        
        if type(datatype) != Datatype and not issubclass(datatype, JsonDumpableMixin):
            raise TypeError(f"Data type argument must be a Datatype enum member, not {datatype}")

        path = None
        try:
            if issubclass(datatype, JsonDumpableMixin):
                dt_extension = 'json'
        except TypeError:
            dt_extension = datatype.value

        if datatype == Datatype.tabular:
            path = os.path.join(self.csv_files_path, f"{key}{dt_extension}")
        elif datatype == Datatype.model:
            # Models may have one of two extensions:
            # .pkl for skorch, or .pth for pytorch. Also,
            # for skorch models the models directory will
            # have an opt.pkl, in addtion to the file
            # whose name matches the key:
            path = None
            for model_file in os.listdir(self.models_path):
                file_p = Path(model_file)
                if file_p.suffix in dt_extension and \
                   file_p.stem == key:
                    path = os.path.join(self.models_path, model_file)
                    break
        elif datatype == Datatype.figure:
            # Figures may have different extensions: png, pdf, etc.:
            path = None
            for fig_file in os.listdir(self.figs_path):
                if Path(fig_file).stem == key:
                    path = os.path.join(self.figs_path, fig_file)
        elif datatype == Datatype.hparams:
            # Allow for a json or a .cfg-type file:
            path = None
            for hparams_file in os.listdir(self.hparams_path):
                if Path(hparams_file).stem == key:
                    path = os.path.join(self.hparams_path, hparams_file)
                
        elif datatype == Datatype.tensorboard:
            path = os.path.join(self.tensorboard_path, f"{key}")
            
        elif type(datatype) == type and issubclass(datatype, JsonDumpableMixin):
            if Path(key).suffix != '.json':
                fname = key + '.json'
            else:
                fname = key
            path = os.path.join(self.json_files_path, fname)
            
        elif datatype == Datatype.txt:
            if Path(key).suffix != '.txt':
                fname = key + '.txt'
            else:
                fname = key
            path = os.path.join(self.txt_files_path, fname)
            
        elif datatype == Datatype.untyped:
            if Path(key).suffix != '.txt':
                fname = key + '.txt'
            else:
                fname = key
            path = os.path.join(self.untyped_files_path, fname)

        return path


    #------------------------------------
    # listdir
    #-------------------
    
    def listdir(self, datatype):
        '''
        Return a list all files of the given datatype
        
        :param datatype: type of files to return
        :type datatype: Datatype
        :return: list of files
        :rtype: [str]
        '''
        
        try:
            storage_dir = self.dir_dict[datatype]
        except KeyError:
            # Could be json-dumpable class:
            try:
                if issubclass(datatype, JsonDumpableMixin):
                    storage_dir = self.dir_dict[JsonDumpableMixin]
                else:
                    raise TypeError(f"Datatype {datatype} not recognized as saved in experiment manager")
            except TypeError:
                raise TypeError(f"Datatype {datatype} not recognized as saved in experiment manager")
        return os.listdir(storage_dir)

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

    #------------------------------------
    # collect_experiment_roots
    #-------------------
    
    @classmethod
    def collect_experiment_roots(cls, all_exps_root, common_time_stamp):
        '''
        Given the root of potentially many experiment directories, and
        the timestamp under which some subset were created from running
        the run_inference.py script, return a dict mapping species to 
        experiment directory name.
        
        The directories will be just the names without the full
        path, which can be recreated easily by prependeing all_exps_root.
        
        Experiment roots look like this:
        
              Classifier_WTROC_2021-09-18T15_00_16_inference

        :param all_exps_root: root of a set of experiment manager
            root subdirectories
        :type all_exps_root: str
        :return: dict mapping each species to the name of 
            an inference subdirectory; may be empty
        :rtype: {str : str}
        '''
        species_date_pattern = re.compile(r"Classifier_([A-Z]{5})_([T0-9-_]{19})_inference")

        dirs = []
        for exp_dir_name in os.listdir(all_exps_root):
            exp_dir = os.path.join(all_exps_root, exp_dir_name)
            dir_name = Path(exp_dir).stem
            if not os.path.isdir(exp_dir) or \
               not dir_name.startswith('Classifier') or \
               not dir_name.endswith('_inference'):
                continue
            match = species_date_pattern.search(dir_name)
            if match is None:
                # Some other type of directory:
                continue
            species, timestamp = match.groups()
            if timestamp != common_time_stamp:
                # It's an experiment from a differenct
                # training:
                continue
            
            # Remember the species and the dir name
            # of the experiment (without parents):
            dirs.append((species, dir_name))
            
        # Sort by species:
        exp_dict = dict(sorted(dirs))
        return exp_dict


    # --------------- Private Methods --------------------

    #------------------------------------
    # _load_model
    #-------------------
    
    def _load_model(self, path, net):
        '''
        Load previously saved weights back
        into a given model. Handles pytorch and
        skorch models.
        
        For skorch models, net.initialize() must
        have been called by the client. For those
        models, the optimizer and history will also 
        be loaded into the model. That is not the case
        for pytorch models.
        
        Distinguishes between the model types by the 
        file exension: .pth for pytorch, .pkl for skorch
        models.
        
        Fun Fact: saved skorch models reportedly use pytorch's
        state_dict() export under the hood. So in theory a
        skorch model should be loadable into a pytorch net.
        
        :param path: path to either .pkl or .pth model file
        :type path: str
        :param net: either pytorch or skorch net
        :type net: {torch.nn.Module | skorch.classifier.NeuralNet}
        :return the net with newly loaded weights
        :rtype {torch.nn.Module | skorch.classifier.NeuralNet}
        '''

        net_path_p = Path(path)
        if isinstance(net, torch.nn.Module):
            # Protect against training having been on GPU, but
            # testing on CPU-only:
            try:
                net.load_state_dict(torch.load(path))
            except RuntimeError:
                if not torch.cuda.is_available():
                    net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        
        elif isinstance(net, skorch.classifier.NeuralNet):
            try:
                # skorch models have extension .pkl, the
                # default model path matches the pytorch convention
                # of .pth. Correct the extension:
                net_path_p = net_path_p.parent.joinpath(net_path_p.stem + '.pkl')
                
                optimizer_path = net_path_p.parent.joinpath('optimizer.pkl')
                history_path   = net_path_p.parent.joinpath('history.json')
                net.load_params(
                    f_params=net_path_p, 
                    f_optimizer=str(optimizer_path),
                    f_history=str(history_path)
                    )
            except RuntimeError:
                if not torch.cuda.is_available():
                    net.load_params(
                        f_params=path, 
                        f_optimizer=str(optimizer_path),
                        f_history=str(history_path)
                        #******* FIND HOW TO RUN NET ON DIFFERENT DEVICE
                        #f_map_location=torch.device('cpu')
                        )
        else:
            raise TypeError(f"The network passed into read() must be a pytorch or skorch net, not {type(net)}")
            
        return net

    #------------------------------------
    # _open_config_files 
    #-------------------
    
    def _open_config_files(self):
        '''
        Finds files in the hparams subdirectory. Any
        files there are assumed to be either json files from
        previously saved NeuralNetConfig instances, or
        configuration files.
        
        (Re)creates NeuralNetConfig instances, and sets value of
        the corresponding keys in the dict API to those instances. 
        Keys are the file names without extension.


        '''
        
        # If the hparams path contains json files of
        # saved configs, turn them into NeuralNetConfig instances,
        # and assign those to self[<keys>] with one key
        # for each hparam json file (usually that will just
        # be one):
        
        for file in os.listdir(self.hparams_path):
            path = os.path.join(self.hparams_path, file)
            # Json file?
            if Path(file).suffix == '.json':
                with open(path, 'r') as fd:
                    config_str = fd.read()
                    config = NeuralNetConfig.json_loads(config_str)
            else:
                # Assumed to be a config file as per Python's
                # configparser syntax:
                config = NeuralNetConfig(path)
            key = Path(path).stem
            # Set a dict-API key/val pair with key
            # equal to the configuration file w/o 
            # extension:
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
                    # Macos sometimes adds weird quarantine files
                    # with a leading underscore; skip those:
                    try:
                        col_names = csv.DictReader(fd).fieldnames
                    except UnicodeDecodeError as _e:
                        continue
            
                # Make the writer:
                fd = open(path, 'a')
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

    def _save_records(self, 
                      item, 
                      fname, 
                      index_col=None, 
                      trust_list_dim=True,
                      header=None):
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
        
        The header argument may be provided the first time any
        data are saved to key.
            
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
        :param header: column names to use as header in CSV file
        :type header: [str] 
        :return full path to the csv file
        :rtype str
        :raise TypeError if item type is unrecognized, or 
            header is provided, but item is not None
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
            if header is None:
                header = self._get_field_names(item, 
                                               index_col=index_col, 
                                               trust_list_dim=trust_list_dim)
            else:
                # Is the index_col header names provided?
                if index_col is not None:
                    header = [index_col] + header
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
                    # Used to be "index_col = None"
                    index_col = index_col

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
                    try:
                        csv_writer.writerow(row_dict)
                    except ValueError as e:
                        # Typical cause of a value error is 
                        # that csv_writer was closed. Add debug
                        # info and re-raise:
                        msg = "While writing csv record: "
                        if type(csv_writer) == csv.DictWriter:
                            msg += f"fname is {csv_writer.fd.name}; "
                        raise ValueError(f"{msg} {repr(e)}") from e
                    
        # Numpy array or Python list:
        elif type(item) in(np.ndarray, list):
            num_dims = len(self._list_shape(item)) if type(item) == list else len(item.shape)
            if num_dims == 1:
                csv_writer.writerow(self._arr_to_dict(item, header))
            else:
                for row in item:
                    csv_writer.writerow(self._arr_to_dict(row, header))

        # A dict:
        elif type(item) == dict:
            # This is a DictWriter's native food:
            csv_writer.writerow(item)

        # If none of the above types, item must be None:
        elif item is not None:
            raise TypeError(f"Unknown item type {item}")
            
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
                config = NeuralNetConfig.json_loads(config_info)
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
    # _is_experiment_path 
    #-------------------
    
    def _is_experiment_path(self, path):
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
    # _experiment_file_name 
    #-------------------
    
    def _experiment_file_name(self, key):
        '''
        If a file exists under the root dir with the
        name key (after extension is removed), method
        returns an absolute path to that file. Else 
        returns None.

        Used when ensuring that a dict key does not conflict with 
        a key that leads to a an experiment file.
         
        :param key: key to examine
        :type key: str
        :return True if file with any extension but
            named the same as key exists under the root
            directory
        :rtype: bool
        '''
        for search_root, _dirs, fnames in os.walk(self.root):
            for fname in fnames:
                if Path(fname).stem == key:
                    return os.path.join(search_root, fname)
        return None


    #------------------------------------
    # _create_dir_if_not_exists 
    #-------------------
    
    def _create_dir_if_not_exists(self, path):
        
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
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
    # update
    #-------------------
    
    def update(self, *args, **kwargs):
        '''
        Save to json every time the dict is changed.
        '''
        super().update(*args, **kwargs)
        self.save()


    #------------------------------------
    # __delitem__
    #-------------------
    
    def __delitem__(self, key):
        
        # Delete the key/val pair:
        super().__delitem__(key)
        self._schedule_save()

    #------------------------------------
    # __repr__
    #-------------------
    
    def __repr__(self):
        root_dir_name = Path(self.root).name
        return f"<ExpMan {root_dir_name} {hex(id(self))}"

    #------------------------------------
    # __hash__ 
    #-------------------
    
    def __hash__(self):
        return id(self)
        

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
        

