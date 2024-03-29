# Machine Learning Experiment Manager
Simply manage pytorch and skorch models, hparams, output charts, and results CSV files

Container to hold all information about an experiment: tabular data
created during training and inference, hyper parameters, charts,
tensorboard entries, and (pytorch/skorch) model snapshots. These data are
stored on disk in standard formats that are accessible by other tools,
such as Excel, pytorch, and image viewers. Except for model snapshots, the
facility is ML platform agnostic. Only pytorch and skorch model instances are
currently handled for saving, though all other features remain fully
functional.

The user associates each asset with a key (string) of their
choice. The manager accesses the asset when given a key.

Selected Details:

- Persistent dict API for 'non-special' data, such as lists,
  numbers, or dicts.
- Uniform API for special machine learning data types like
    tables, pyplot figures, and model snapshots.
- Creates human-readable files under a single directory root
- Entire experiment archives are movable/copyable with standard OS tools.
- Files on disk are .csv, for tables, Pandas Series and DataFrame
  instances, pdf, png, etc. for pyplot figures, state_dict .pth
  files for pytorch models, and .pkl save_params exports of skorch
  models.

# Starting an Experiment and Saving Data

Examples: an experiment instance is started to retain all data in
subdirectories of MyExperiment under the current directory. All
archived data will reside under this root.

'''
    pip install ml-experiment-manager
'''

The primary method is `save()`, which accepts an arbitrary string
**key**, and data of varying types. The method stores the data in
files appropriate for those types:
  
```
    from experiment_manager.experiment_manager import ExperimentManager, Datatype
    
    experiment_archive = os.path.join(os.path.dirname(__file__), 'MyExperiment')
    exp = ExperimentManager(experiment_archive)

    # Create a tabular entry called 'my_tbl'. An underlying .csv file
    # named my_tbl.csv will be created under experiment_archive:

    probabilities = pd.DataFrame([[.1,.2,.7],[.4,.5,.1]], columns=['foo', 'bar'])
    exp.save('my_tbl', probabilities)

    # Add a row to the same tabular content (i.e. to the same .csv file):
    exp.save('my_tbl, pd.Series([.7,.1,.2], index=['foo', 'bar']))

    # A bit more data still
    df_more = pd.DataFrame([[.3,.3,.4],[.5,.5.,0],[.6,.2,.2]],
                           columns=['foo', 'bar'])
    exp.save('my_tbl', df_more)

    # Save a pytorch or skorch model snapshot; the manager will save the state_dict,
    # using torch.save() or model.save_params() for skorch models. The type of
    # model is detected automatically:
    # 
    exp.save('model_snaphot1', model1)

    # Save another snapshot
    exp.save('model_snaphot2', model2)

    # Result chart figures in format indicated by
    # extension of the archive key:

    exp.save('pr_figure.pdf', pr_curve)
    exp.save('cm.png', conf_matrix)

    # A tensorboard location. Note that in this case
    # only the key is provided. This string will be a subdirectory
    # name under the experiment root. As usual for the
    # full path to that directory is returned, and can be
    # used when creating the tensorboard writer:
    
    exp.save('tb_data')
    
    # Saving any other, non-special type of data, as long as
    # json.dump() can handle it: use ExperimentManager instances like
    # a persistent dict. The new dict state is saved soon after being
    # updated

    exp['some_number'] = 10
    exp['some_other_number'] = 20
```

Client's should call `close()` on an experiment to close all open
writers. 

# Opening Existing Experiments

To reopen an existing experiment:

```
    exp = ExperimentManager(experiment_archive_root)
```

The data saved in the experiment can be retrieved as appropriate
Python data types through the `read()` method. Callers must provide
the data key and the type of data being requested.

```
   # Obtain a Pandas DataFrame instance:
   my_tbl_df = exp.read('my_tbl', Datatype.tabular)

   # A pyplot Figure instance:
   my_fig    = exp.read('pr_figure.pdf', Datatype.figure)

   # Pytorch and skorch model retrievals work by
   # creating an uninitialized model, i.e. a pytorch module,
   # or a skorch net. That model is passed to read, which
   # returns the model with state initialized.
   my_model = <create torch.nn.Module or skorch.classifier.NeuralNet>
   my_model_initialized  = exp.read('model_snaphot1', Datatype.model, initialized_net=my_model)

   # A [hyperparameter configuration][#hyperparameters]:
   config = exp.read('hparams')

   # The full tensorboard directory path:
   tb_path = exp.read('tb_data')
```


# Hyperparameters

Hyperparameter values may be stored as dict key/value pairs
(`exp['lr'] = 0.8`). However a class NeuralNetConfig is available for
organizing hyperparameters. Given an instance `config` of this class,

```
    exp.save('hparams', config)
```
will create a copy of the configuration under the experiment root.

`NeuralNetConfig`  extends the [standard Python `configparser`
package](https://docs.python.org/3/library/configparser.html).  That
is, NeuralNetConfig instances read configuration files of the form

```
[Paths]

# Root of the data/test files:
root_train_test_data = /Users/paepcke/EclipseWorkspacesNew/birds/src/birdsong/tests/data/birds

[Training]

net_type      = resnet18
batch_size    = 64
lr            = 0.8
pretrained    = True
class_names   = foo,bar,baz

       etc.
```

where `Path` and `Training` are called *sections*. The NeuralNetConfig
class adds the following convenience methods:

```
   # Obtain specific data types, rather than strings:

   config.getint('Training', 'batch_size')
   config.getfloat('Training', 'lr')
   config.getboolean('Training', 'pretrained')
   config.getarray('Training', 'class_names')
   config.sections()

   config.copy()

   # Equality test:
   config1 == config2
   
   config.to_json()
   config.from_json()
```

# Miscellaneous Methods

```
   # Obtain full path to any saved data;
   # datatype is an element of the Datatype
   # enumeration:
   exp.abspath(<key>, datatype)
```

