'''
Created on Aug 5, 2021

@author: paepcke
'''
import csv
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import zlib

from experiment_manager.experiment_manager import ExperimentManager, AutoSaveThread, Datatype, \
    JsonDumpableMixin, TypeConverter
from experiment_manager.neural_net_config import NeuralNetConfig
import skorch
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#TEST_ALL = True
TEST_ALL = False

'''
TODO:
   o test moving the experiment: ensure relative addressing!
'''

class Jsonable(JsonDumpableMixin):
    def __init__(self):
        self.my_dict = {'key1' : 'The goose',
                        'key2' : 'is cooked'
                       }
    def json_dump(self, fname):
        with open(fname, 'w') as fd:
            json.dump(json.dumps(self.my_dict), fd)
    @classmethod
    def json_load(cls, fname):
        my_dict = json.loads(json.load(fname))
        obj = Jsonable()
        obj.my_dict = my_dict
        return obj 

class ExperimentManagerTest(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(__file__)
        cls.exp_fname = 'experiment'
        cls.prefab_exp_fname = 'fake_experiment'
        
        cls.exp_root = os.path.join(cls.curr_dir, cls.exp_fname)
        cls.prefab_exp_root = os.path.join(cls.curr_dir, cls.prefab_exp_fname)
    
    def setUp(self):
        
        try:
            shutil.rmtree(self.prefab_exp_root)
        except FileNotFoundError:
            pass

        os.makedirs(self.prefab_exp_root)
        
        # Create a little torch model and save it:
        models_dir = os.path.join(self.prefab_exp_root,'models')
        os.makedirs(models_dir)
        model_path = os.path.join(models_dir, 'tiny_model.pth')
        self.tiny_model = TinyModel()
        torch.save(self.tiny_model.state_dict(), model_path)
        
        # Same for a skorch model:
        self.tiny_skorch = skorch.classifier.NeuralNetBinaryClassifier(TinyModel)
        self.tiny_skorch.initialize()
        self.skorch_model_path = os.path.join(models_dir, 'tiny_skorch.pkl')
        self.skorch_opt_path   = os.path.join(models_dir, 'optimizer.pkl')
        self.skorch_hist_path  = os.path.join(models_dir, 'history.json')
        self.tiny_skorch.save_params(f_params=self.skorch_model_path, 
                                     f_optimizer=self.skorch_opt_path, 
                                     f_history=self.skorch_hist_path)
        
        # Create two little csv files:
        csvs_dir   = os.path.join(self.prefab_exp_root,'csv_files')
        os.makedirs(csvs_dir)
        self.make_csv_files(csvs_dir)
        
        # Create a little json file:
        json_dir   = os.path.join(self.prefab_exp_root,'json_files')
        os.makedirs(json_dir)
        self.make_json_file(json_dir)
        
        # Create some untyped files
        untyped_dir = os.path.join(self.prefab_exp_root,'untyped_files')
        os.makedirs(untyped_dir)
        self.make_untyped_files(untyped_dir)

        # Create a tiny png file:
        figs_dir   = os.path.join(self.prefab_exp_root,'figs')
        os.makedirs(figs_dir)
        with open(os.path.join(figs_dir, "tiny_png.png") ,"wb") as fd:
            fd.write(self.makeGrayPNG([[0,255,0],[255,255,255],[0,255,0]]))
            
        hparams_dir = os.path.join(self.prefab_exp_root,'hparams')
        os.makedirs(hparams_dir)
        self.hparams_path = self.make_neural_net_config_file(hparams_dir)
        
    def tearDown(self):
        try:
            self.exp.close()
        except:
            pass
        try:
            shutil.rmtree(self.exp_root)
        except:
            pass
        try:
            shutil.rmtree(self.prefab_exp_root)
        except:
            pass

# ------------------- Tests --------------

    #------------------------------------
    # test_creation
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_creation(self):
        exp = ExperimentManager(self.exp_root)
        self.assertEqual(exp['root_path'], self.exp_root)
        self.assertEqual(exp['_models_path'], os.path.join(self.exp_root, 'models'))
        self.assertTrue(exp.csv_writers == {})
        
        # Should have a json file in root dir:
        self.assertTrue(os.path.exists(os.path.join(self.exp_root, 'experiment.json')))
        
        # Delete and restore the experiment:
        exp.close()
        del exp
        
        exp1 = ExperimentManager(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp1
        self.assertEqual(exp1['root_path'], self.exp_root)
        self.assertEqual(exp1['_models_path'], os.path.join(self.exp_root, 'models'))
        self.assertTrue(exp1.csv_writers == {})

    #------------------------------------
    # test_dict_addition 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_dict_addition(self):
        
        exp = ExperimentManager(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp

        tst_dict = {'foo' : 10, 'bar' : 20}
        csv_file_path = exp.save('first_dict', tst_dict)
        
        with open(csv_file_path, 'r') as fd:
            reader = csv.DictReader(fd)
            self.assertEqual(reader.fieldnames, ['foo', 'bar'])
            row_dict = next(reader)
            self.assertEqual(list(row_dict.values()), ['10','20'])
            self.assertEqual(list(row_dict.keys()), ['foo', 'bar'])

        writers_dict = exp.csv_writers
        self.assertEqual(len(writers_dict), 1)
        
        wd_keys   = list(writers_dict.keys())
        first_key = wd_keys[0]
        self.assertEqual(first_key, Path(csv_file_path).stem)
        self.assertEqual(type(writers_dict[first_key]), csv.DictWriter)

        # Add second row to the same csv:
        row2_dict = {'foo' : 100, 'bar' : 200}
        exp.save('first_dict', row2_dict)
        
        # Second row should be [100, 200]:
        with open(csv_file_path, 'r') as fd:
            reader = csv.DictReader(fd)
            row_dict0 = next(reader)
            self.assertEqual(list(row_dict0.values()), ['10','20'])
            row_dict1 = next(reader)
            self.assertEqual(list(row_dict1.values()), ['100','200'])

        # Should be able to just write a row, not a dict:
        exp.save('first_dict', [1000,2000])
        # Look at 3rd row should be ['1000', '2000']:
        with open(csv_file_path, 'r') as fd:
            reader = csv.DictReader(fd)
            row_dict0 = next(reader)
            self.assertEqual(list(row_dict0.values()), ['10','20'])
            row_dict1 = next(reader)
            self.assertEqual(list(row_dict1.values()), ['100','200'])
            row_dict2 = next(reader)
            self.assertEqual(list(row_dict2.values()), ['1000','2000'])
        
    #------------------------------------
    # test_saving_csv
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_saving_csv(self):
        exp = ExperimentManager(self.exp_root)

        tst_dict = {'foo' : 10, 'bar' : 20}
        csv_file_path = exp.save('first_dict', tst_dict)
        exp.close()
        
        del exp
        
        # Reconstitute the same experiment:
        exp = ExperimentManager(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp
        
        # First, ensure that the test dict 
        # is unharmed without using the ExperimentManager
        # instance:
        with open(csv_file_path, 'r') as fd:
            reader = csv.DictReader(fd)
            self.assertEqual(reader.fieldnames, ['foo', 'bar'])
            row_dict = next(reader)
            self.assertEqual(list(row_dict.values()), ['10','20'])
            self.assertEqual(list(row_dict.keys()), ['foo', 'bar'])

        # Now treat the experiment
        writers_dict = exp.csv_writers
        self.assertEqual(len(writers_dict), 1)

    #------------------------------------
    # test_csv_header_first
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_csv_header_first(self):

        exp = ExperimentManager(self.exp_root)
        self.exp = exp

        csv_file_path = exp.save('header_first', item=None, header=['foo', 'bar'])

        tst_list = [[1,2],[3,4]]
        exp.save('header_first', tst_list)
        exp.close()
        
        with open(csv_file_path, 'r') as fd:
            reader = csv.DictReader(fd)
            first_row_dict = next(reader)
            expected = {'foo' : '1', 'bar' : '2'}
            self.assertDictEqual(first_row_dict, expected)
            
            second_row_dict = next(reader)
            expected = {'foo' : '3', 'bar' : '4'}
            self.assertDictEqual(second_row_dict, expected)

    #------------------------------------
    # test_adding_to_csv_index_ignored
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_adding_to_csv_index_ignored(self):
        
        exp = ExperimentManager(self.exp_root)
        self.exp = exp
        
        tlist = [[1,3,4], [4,5,6]]
        self.exp.save('my_list', tlist)
        exp.close()
        
        csv_path = self.exp.abspath('my_list', Datatype.tabular)
        
        retrieved = []
        with open(csv_path, 'r') as fd:
            reader = csv.reader(fd)
            for line in reader:
                retrieved.append(line)
                
        expected = [['0', '1', '2'], ['1', '3', '4'], ['4', '5', '6']]
        self.assertListEqual(retrieved, expected)
        
        re_read  = self.exp.read('my_list', Datatype.tabular)
        self.assertListEqual(re_read, tlist)

        df = pd.DataFrame([[1,2,3],[10,20,30]], index=[(3, 0.5, 0.01), (4, 0.6, 0.08)], columns=[100,200,300])
        exp.save('df', df)
        exp.close()

    #------------------------------------
    # test_adding_to_csv_index_included
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_adding_to_csv_index_included(self):
        
        exp = ExperimentManager(self.exp_root)
        self.exp = exp
        
        df = pd.DataFrame([[1,2,3],[10,20,30]], index=[(3, 0.5, 0.01), (4, 0.6, 0.08)], columns=[100,200,300])
        exp.save('df', df, index_col='the_idx')
        exp.close()
        
        row_dicts = self.read_csv_file('df')
        expected  = [{'the_idx': '(3, 0.5, 0.01)', '100': '1', '200': '2', '300': '3'},
                     {'the_idx': '(4, 0.6, 0.08)', '100': '10', '200': '20', '300': '30'}]
        for i, one_dict in enumerate(row_dicts):
            self.assertDictEqual(one_dict, expected[i])

        exp = ExperimentManager(self.exp_root)
        exp.save('df', df)
        row_dicts = self.read_csv_file('df')
        expected  = [{'the_idx': '(3, 0.5, 0.01)', '100': '1', '200': '2', '300': '3'},
                     {'the_idx': '(4, 0.6, 0.08)', '100': '10', '200': '20', '300': '30'},
                     {'the_idx': '(3, 0.5, 0.01)', '100': '1', '200': '2', '300': '3'},
                     {'the_idx': '(4, 0.6, 0.08)', '100': '10', '200': '20', '300': '30'}
                     ]

        for i, one_dict in enumerate(row_dicts):
            self.assertDictEqual(one_dict, expected[i])

    #------------------------------------
    # test_type_converter
    #-------------------
    
    #******* Run all tests, and keep debuggin
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_type_converter(self):
        
        converter = TypeConverter()
        
        l = [1,2,3]
        l_str = converter(l, 'str')
        expected = ['1', '2', '3']
        self.assertListEqual(l_str, expected)

        l = [[1,2,3], [4,5,6]]
        l_str = converter(l, 'str')
        expected = [['1', '2', '3'],['4', '5', '6']]
        self.assertListEqual(l_str, expected)
    
        l = [[1,2,3], [4,5,6]]
        l_str = converter(l, 'float')
        expected = [[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]]
        self.assertListEqual(l_str, expected)
    
        a = np.array([1,2,3])
        a_float = converter(a, 'float')
        expected = np.array([1.0, 2.0, 3.0])
        self.assertTrue((a_float == expected).all())
        
        a = np.array([[1,2,3], [4,5,6]])
        a_float = converter(a, 'float')
        expected = np.array([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
        self.assertTrue((a_float == expected).all())
        
        t = (1,2,3)
        l = converter(t, 'list')
        expected = [1,2,3]
        self.assertListEqual(l, expected)
        
        t = set([1,2,3])
        l = converter(t, 'list')
        expected = [1,2,3]
        self.assertListEqual(l, expected)
        
        s = set([1,2,3])
        s_floats = converter(s, 'float')
        expected = set([1.0, 2.0, 3.0])
        self.assertSetEqual(s_floats, expected)

    #------------------------------------
    # test_saving_hparams
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_saving_hparams(self):
        
        exp = ExperimentManager(self.exp_root)

        exp.add_hparams('my_config', self.hparams_path)
        config_obj = exp['my_config']
                         
        # Should have a json export of the config instance:
        saved_copy_path = os.path.join(exp.hparams_path, 'my_config.json')
        with open(saved_copy_path, 'r') as fd:
            json_str = fd.read()
            other_config_obj = NeuralNetConfig.json_loads(json_str)
            # Couple of spot checks that the config instance
            # behaves as expected:
            self.assertEqual(other_config_obj['Training']['net_name'], 'resnet18')
            self.assertEqual(other_config_obj.getint('Parallelism', 'master_port'), 5678)

        # The config instance should be available
        # via the config key:
        self.assertEqual(config_obj, exp['my_config'])
        # Couple of spot checks that the config instance
        # behaves as expected:
        self.assertEqual(config_obj['Training']['net_name'], 'resnet18')
        self.assertEqual(config_obj.getint('Parallelism', 'master_port'), 5678)

        exp.close()
        
        del exp
        
        # Reconstitute the same experiment:
        exp1 = ExperimentManager(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp1
        
        config_obj = exp1['my_config']
        self.assertEqual(config_obj['Training']['net_name'], 'resnet18')
        self.assertEqual(config_obj.getint('Parallelism', 'master_port'), 5678)

    #------------------------------------
    # test_saving_json
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_saving_json(self):
        
        exp = ExperimentManager(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp
        my_jsonable = Jsonable()
        
        dst = exp.save('jsonable_test', my_jsonable)
        # Read the raw json:
        with open(dst, 'r') as fd:
            jstr = json.loads(fd.read())
            recovered_dict = json.loads(jstr)
            self.assertDictEqual(recovered_dict, my_jsonable.my_dict)

        exp.close()

    #------------------------------------
    # test_saving_txt
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_saving_txt(self):

        exp = ExperimentManager(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp
        readme = "This is my README"
        
        exp.save('my_readme', readme)
        with open(os.path.join(exp.txt_files_path, 'my_readme.txt'), 'r') as fd:
            recovered = fd.read()
            self.assertEqual(recovered, readme)

    #------------------------------------
    # test_mixed_csv_adding 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_mixed_csv_adding(self):

        exp = ExperimentManager(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp
        
        # Start with saving a dict:
        my_dict = {'foo' : 1, 'bar' : 2}
        exp.save('my_results', my_dict)
        
        # Add a data frame to the same csv:
        df = pd.DataFrame([[3,4],[5,6],[7,8]], columns=['foo', 'bar'])
        csv_path = exp.save('my_results', df)
        
        rows = self.read_ints_from_csv(csv_path)

        expected = [[1,2],
                    [3,4],
                    [5,6],
                    [7,8]
                    ]
        self.assertListEqual(rows, expected)
        
        # Add a row in the form of a pd Series:
        ser = pd.Series([9,10], index=['foo', 'bar'])
        exp.save('my_results', ser)
        expected.append([9,10])
        rows = self.read_ints_from_csv(csv_path)
        self.assertListEqual(rows, expected)
        
        # Add an np array:
        nparr = np.array([[11,12],[13,14]])
        exp.save('my_results', nparr)
        expected.extend([[11,12],[13,14]])
        rows = self.read_ints_from_csv(csv_path)
        self.assertListEqual(rows, expected)

    #------------------------------------
    # test_saving_mem_items
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_saving_mem_items(self):
        
        exp = ExperimentManager(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp
        
        my_dict = {'foo' : 10, 'bar' : 20}

        # Get the current json representation of
        # the experiment from disk:
        with open(os.path.join(exp.root, 'experiment.json'), 'r') as fd:
            exp_json = json.load(fd)
            
        try:
            exp_json['my_dic']
            self.fail("Should have received KeyError")
        except KeyError:
            # Good! Shouldn't have my_dict yet:
            pass

        # Set the in-memory dict value:
        exp['my_dict'] = my_dict
        
        # Make sure it took:
        self.assertDictEqual(exp['my_dict'], my_dict)
        
        # Get a second experiment manager pointing 
        # to the same experiment dir tree (not a usual
        # situation, but could happen).
        # Make sure exp is not in the middle of saving.
        # Not something regular user needs to do. Necessary
        # here b/c we are loading a second experiment to
        # the *same* experiment that is already active:
        exp.auto_save_thread.join()
        exp1 = ExperimentManager(self.exp_root)
        
        # The exp should still have my_dict in memory:
        self.assertDictEqual(exp['my_dict'], my_dict)

        # Set the 'my_dict' key in exp1 to a new dict:
        animal_dict = {'horse' : 'big', 'mouse' : 'small'}
        exp1['my_dict'] = animal_dict
        
        # And *still*, the exp in-memory version would not
        # have seen this change, b/c there is no automatic
        # refresh from the json representation on disk: 
        self.assertDictEqual(exp['my_dict'], my_dict)
        # But exp1 should see the assignment:
        self.assertDictEqual(exp1['my_dict'], animal_dict)

        # After reloading exp, the value should change.
        # But we have to give the auto-save a chance first.
        # This is not something regular users would need
        # to do:
        exp.auto_save_thread.join()
        exp1.auto_save_thread.join()
        
        self.assertFalse(exp.auto_save_thread.is_alive())
        self.assertFalse(exp1.auto_save_thread.is_alive())
        
        exp = ExperimentManager(self.exp_root)
        self.assertDictEqual(exp['my_dict'], animal_dict)
        
        exp.auto_save_thread.cancel()

    #------------------------------------
    # test_destroy
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_destroy(self):
        
        exp = ExperimentManager(self.exp_root)

        exp.save('foo', pd.Series([1,2,3], index=['blue', 'green', 'yellow']))
        expected_csv_path  = os.path.join(exp.csv_files_path, 'foo.csv')
        expected_json_path = os.path.join(exp.csv_files_path, 'foo.json')
        self.assertTrue(os.path.exists(expected_csv_path))
        self.assertTrue(os.path.exists(expected_json_path))
        
        # Now delete the file:
        exp.destroy('foo', Datatype.tabular)
        self.assertFalse(os.path.exists(expected_csv_path))
        self.assertFalse(os.path.exists(expected_json_path))

        # Same for Figure:
        fig = plt.Figure()

        exp.save('bar', fig)
        expected_path = os.path.join(exp.figs_path, 'bar.pdf')
        self.assertTrue(os.path.exists(expected_path))
        
        exp.destroy('bar', Datatype.figure)
        self.assertFalse(os.path.exists(expected_path))

        # Same with tensorboard info, which should clear
        # out a tensorboard directory:
        exp.save('my_tensorboard')
        expected_path = os.path.join(exp.tensorboard_path, 'my_tensorboard')
        self.assertTrue(os.path.exists(expected_path) and os.path.isdir(expected_path))
        self.assertTrue(len(os.listdir(exp.tensorboard_path)) == 1)
        
        exp.destroy('my_tensorboard', Datatype.tensorboard)
        self.assertTrue(os.path.exists(exp.tensorboard_path))
        self.assertTrue(len(os.listdir(exp.tensorboard_path)) == 0)
        
        # Json:
        jobj = Jsonable()
        exp.save('my_json', jobj)
        expected_path = os.path.join(exp.json_files_path, 'my_json.json')
        self.assertTrue(os.path.exists(expected_path))
        
        exp.destroy('my_json', Jsonable)
        self.assertFalse(os.path.exists(expected_path))
        
        # Text:
        readme = "My README."
        exp.save('my_readme', readme)
        expected_path = os.path.join(exp.txt_files_path, 'my_readme.txt')
        self.assertTrue(os.path.exists(expected_path))
        
        exp.destroy('my_readme', Datatype.txt)
        self.assertFalse(os.path.exists(expected_path))
        

    #------------------------------------
    # test_root_movability
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_root_movability(self):
        exp = ExperimentManager(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp
        exp['foo'] = 10
        index_strs = ['row0', 'row1']
        col_strs   = ['foo', 'bar', 'fum']
        tst_df = pd.DataFrame([[1,2,3], [4,5,6]], columns=col_strs, index=index_strs)
        exp.save('my_df', tst_df)
        config = NeuralNetConfig(self.hparams_path)
        exp.save('my_config', config)

        exp.close()
        del exp
        
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='exp_man_tests') as tmp_dir_name:
            shutil.move(self.exp_root, tmp_dir_name)
            new_root = os.path.join(tmp_dir_name, Path(self.exp_root).stem)
            exp1 = ExperimentManager(new_root)
            
            self.assertEqual(exp1['foo'], 10)
            
            df_path = exp1.abspath('my_df', Datatype.tabular)
            _recovered_df = pd.read_csv(df_path)
            #print(recovered_df)
            
            recovered_config = exp1['my_config']
            self.assertDictEqual(recovered_config, config)

            exp1.close()
            shutil.move(new_root, Path(self.exp_root).parent)

    #------------------------------------
    # test_abspath
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_abspath(self):
        
        # Pre-made experiment tree with some csv files,
        # a model, and a figure:
        exp = ExperimentManager(self.prefab_exp_root)
        # For cleanup in tearDown():
        self.exp = exp

        tst_dict = {'foo' : 10, 'bar' : 20}
        # Abspath result of a saved dict must match
        # the path returned by the save process:
        csv_file_path = exp.save('first_dict', tst_dict)
        self.assertEqual(exp.abspath('first_dict', Datatype.tabular), csv_file_path)
        
        # Previously saved csv file, just for fun:
        self.assertEqual(exp.abspath('tiny_csv1', Datatype.tabular),
                         os.path.join(exp.csv_files_path, 'tiny_csv1.csv'))
        
        # Data model:
        self.assertEqual(exp.abspath('tiny_model', Datatype.model), 
                         os.path.join(exp.models_path, 'tiny_model.pth'))
        
        # Figure:
        self.assertEqual(exp.abspath('tiny_png', Datatype.figure), 
                         os.path.join(exp.figs_path, 'tiny_png.png'))

        # H-parameters
        exp.add_hparams('my_hparams', self.hparams_path)
        self.assertEqual(exp.abspath('my_hparams', Datatype.hparams), 
                         os.path.join(exp.hparams_path, 'my_hparams.json'))

        # Tensorboard    the-item (a dir name)   the-key
        _dst = exp.save('my_tensorboard')
        tb_path = exp.abspath('my_tensorboard', Datatype.tensorboard)
        self.assertEqual(tb_path, os.path.join(exp.tensorboard_path, 'my_tensorboard'))
        
        # Json:
        self.assertEqual(exp.abspath('tiny_json', Jsonable),
                         os.path.join(exp.json_files_path, 'tiny_json.json'))
        
        # Untyped:
        self.assertEqual(exp.abspath('one_text.txt', Datatype.untyped),
                         os.path.join(exp.untyped_files_path, 'one_text.txt'))
        self.assertEqual(exp.abspath('one_list', Datatype.untyped),
                         os.path.join(exp.untyped_files_path, 'one_list.txt'))
        self.assertEqual(exp.abspath('one_dict', Datatype.untyped),
                         os.path.join(exp.untyped_files_path, 'one_dict.txt'))

    #------------------------------------
    # test_read
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_read(self):
        
        # Pre-made experiment tree with some csv files,
        # a model, and a figure:
        exp = ExperimentManager(self.prefab_exp_root)
        # For cleanup in tearDown():
        self.exp = exp
        
        # Test reading csv files:
        df_csv1_expected = pd.read_csv(self.csv1)
        df_csv1 = exp.read('tiny_csv1', Datatype.tabular)
        self.assertDataframesEqual(df_csv1, df_csv1_expected)
        
        # Test loading state dict into existing pytorch model:
        model_path = os.path.join(exp.models_path, 'tiny_model.pth')
        new_tiny_model = TinyModel()
        new_tiny_model.load_state_dict(torch.load(model_path))
        model = exp.read('tiny_model', 
                         Datatype.model, 
                         uninitialized_net=self.tiny_model)
        self.assertPytorchModelsEqual(model, new_tiny_model)
        
        # Test loading state dict into existing skorch model:
        new_tiny_skorch_model = skorch.classifier.NeuralNetBinaryClassifier(TinyModel)
        new_tiny_skorch_model.initialize()
        new_tiny_skorch_model.load_params(f_params=self.skorch_model_path,
                                          f_optimizer=self.skorch_opt_path,
                                          f_history=self.skorch_hist_path)
        skorch_model = exp.read('tiny_skorch', 
                                Datatype.model, 
                                uninitialized_net=self.tiny_skorch)
        self.assertSkorchModelsEqual(skorch_model, new_tiny_skorch_model)

        
        fig_path = os.path.join(exp.figs_path, 'tiny_png.png')
        fig_expected = plt.imread(fig_path)
        fig = exp.read('tiny_png', Datatype.figure)
        self.assertTrue((fig == fig_expected).all())
        
        hparams_expected = NeuralNetConfig(self.hparams_path)
        # File we created in setup is config.cfg:
        hparams = exp.read('config', Datatype.hparams)
        self.assertEqual(hparams, hparams_expected)
        
        # Case when the hparams are stored as a json file:
        exp.save('config_json', hparams)
        hparams_from_json = exp.read('config_json', Datatype.hparams)
        self.assertEqual(hparams_from_json, hparams_expected)
        
        # Tensorboard: should just return the file name:
        _dst = exp.save('my_tensorboard')
        tb_expected  = os.path.join(exp.tensorboard_path, 'my_tensorboard')
        tb_from_read = exp.read('my_tensorboard', Datatype.tensorboard)
        self.assertEqual(tb_from_read, tb_expected)

    #------------------------------------
    # test_AutoSaveThread
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_AutoSaveThread(self):
        
        self.was_called = False
        thread = AutoSaveThread(self.set_true, 'mandatory is given', optional=20)
        thread.start()
        thread.join()
        self.assertTrue(self.was_called)
        self.assertEqual(self.mandatory, 'mandatory is given')
        self.assertEqual(self.optional, 20)

    #------------------------------------
    # test_get_fieldnames
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_get_fieldnames(self):
        
        # Getting field names from different data structures:
        exp = ExperimentManager(self.exp_root)
        
        # Dataframes including index
        item = pd.DataFrame([[1,2,3],[4,5,6]], 
                            index=['row0', 'row1'], 
                            columns=['foo', 'bar', 'fum'])

        # Dataframes not including index:
        fld_names = exp._get_field_names(item)
        self.assertEqual(fld_names, ['foo', 'bar', 'fum'])
        
        # pd.Series with given index:
        ser = pd.Series([1,2,3], ['foo', 'bar', 'fum'])
        fld_names = exp._get_field_names(ser)
        self.assertEqual(fld_names, ['foo', 'bar', 'fum'])
        
        # pd.Series without index:
        ser = pd.Series([1,2,3])
        fld_names = exp._get_field_names(ser)
        self.assertEqual(fld_names, ['0', '1', '2'])
        
        # dict:
        my_dict = {'foo': 10, 'bar' : 20, 'fum' : 30}
        fld_names = exp._get_field_names(my_dict)
        self.assertEqual(fld_names, ['foo', 'bar', 'fum'])
        
        # numpy arr:
        nparr = np.array([[1,2,3], [4,5,6]])
        fld_names = exp._get_field_names(nparr)
        self.assertEqual(fld_names, ['0', '1', '2'])
        
        l = [[1,2,3], [4,5,6]]
        fld_names = exp._get_field_names(l)
        self.assertEqual(fld_names, ['0', '1', '2'])

        l1D = [1,2,3]
        fld_names = exp._get_field_names(l1D)
        self.assertEqual(fld_names, ['0', '1', '2'])
        
        # List with unequal rows, not trusting:
        l = [[1,2,3], [4,6]]
        try:
            fld_names = exp._get_field_names(l, trust_list_dim=False)
            raise self.fail("Expected ValueError for unequal rows in list")
        except ValueError:
            pass

# -------------------- Utilities --------------


    #------------------------------------
    # set_true
    #-------------------

    def set_true(self, mandatory, optional=10):
        '''
        Used in testing AutoSaveThread
        '''
        self.mandatory = mandatory
        self.optional  = optional
        self.was_called = True

    #------------------------------------
    # make_csv_files
    #-------------------

    def make_csv_files(self, dst_dir):
        '''
        Create example csv files
        
        :param dst_dir:
        :type dst_dir:
        '''
        csv1 = os.path.join(dst_dir, 'tiny_csv1.csv')
        csv2 = os.path.join(dst_dir, 'tiny_csv2.csv')
        
        with open(csv1, 'w') as fd:
            writer = csv.DictWriter(fd, fieldnames=['foo', 'bar'])
            writer.writeheader()
            writer.writerow({'foo' : 10, 'bar' : 20})
            writer.writerow({'foo' : 100,'bar' : 200})
            
        with open(csv2, 'w') as fd:
            writer = csv.DictWriter(fd, fieldnames=['blue', 'green'])
            writer.writeheader()
            writer.writerow({'blue' : 'sky', 'green' : 'grass'})
            writer.writerow({'blue' : 'umbrella', 'green' : 'forest'})
            
        self.csv1 = csv1
        self.csv2 = csv2

    #------------------------------------
    # make_json_file
    #-------------------
    
    def make_json_file(self, dst_dir):
        '''
        Create a json file in the json directory:
        
        :param dst_dir: the json dir
        :type dst_dir: src
        '''
        obj = Jsonable()
        fname = os.path.join(dst_dir, 'tiny_json.json')
        obj.json_dump(fname)

    #------------------------------------
    # make_untyped_files
    #-------------------
    
    def make_untyped_files(self, dst_dir):
        
        with open(os.path.join(dst_dir, 'one_text.txt'), 'w') as fd:
            fd.write("The goose is cooked")
            
        with open(os.path.join(dst_dir, 'one_dict.txt'), 'w') as fd:
            fd.write(str({'foo' : 10, 'bar' : 20}))

        with open(os.path.join(dst_dir, 'one_list.txt'), 'w') as fd:
            fd.write(str([10, 20]))

    #------------------------------------
    # makeGrayPNG
    #-------------------

    def makeGrayPNG(self, data, height = None, width = None):
        '''
        From  https://stackoverflow.com/questions/8554282/creating-a-png-file-in-python
        
        Converts a list of list into gray-scale PNG image.
        __copyright__ = "Copyright (C) 2014 Guido Draheim"
        __licence__ = "Public Domain"
        
        Usage:
            with open("/tmp/tiny_png.png","wb") as f:
                f.write(makeGrayPNG([[0,255,0],[255,255,255],[0,255,0]]))
        
        :return a png structure
        '''
        
        def I1(value):
            return struct.pack("!B", value & (2**8-1))
        def I4(value):
            return struct.pack("!I", value & (2**32-1))
        # compute width&height from data if not explicit
        if height is None:
            height = len(data) # rows
        if width is None:
            width = 0
            for row in data:
                if width < len(row):
                    width = len(row)
        # generate these chunks depending on image type
        makeIHDR = True
        makeIDAT = True
        makeIEND = True
        png = b"\x89" + "PNG\r\n\x1A\n".encode('ascii')
        if makeIHDR:
            colortype = 0 # true gray image (no palette)
            bitdepth = 8 # with one byte per pixel (0..255)
            compression = 0 # zlib (no choice here)
            filtertype = 0 # adaptive (each scanline seperately)
            interlaced = 0 # no
            IHDR = I4(width) + I4(height) + I1(bitdepth)
            IHDR += I1(colortype) + I1(compression)
            IHDR += I1(filtertype) + I1(interlaced)
            block = "IHDR".encode('ascii') + IHDR
            png += I4(len(IHDR)) + block + I4(zlib.crc32(block))
        if makeIDAT:
            raw = b""
            for y in range(height):
                raw += b"\0" # no filter for this scanline
                for x in range(width):
                    c = b"\0" # default black pixel
                    if y < len(data) and x < len(data[y]):
                        c = I1(data[y][x])
                    raw += c
            compressor = zlib.compressobj()
            compressed = compressor.compress(raw)
            compressed += compressor.flush() #!!
            block = "IDAT".encode('ascii') + compressed
            png += I4(len(compressed)) + block + I4(zlib.crc32(block))
        if makeIEND:
            block = "IEND".encode('ascii')
            png += I4(0) + block + I4(zlib.crc32(block))
        return png

    #------------------------------------
    # make_neural_net_config_file
    #-------------------
    
    def make_neural_net_config_file(self, dst_dir):
        '''
        Create a fake neural net configurations file.
        
        :param dst_dir: where to put the 'config.cfg' file
        :type dst_dir: src
        :return full path to new config file
        :rtype src
        '''
        txt = '''
            [Paths]
            
            # Root of the data/test files:
            root_train_test_data = /foo/bar
            
            [Training]
            
            net_name      = resnet18
            # Some comment
            pretrained     = True
            freeze         = 0
            min_epochs    = 2
            max_epochs    = 2
            batch_size    = 2
            num_folds     = 2
            opt_name      = Adam
            loss_fn       = CrossEntropyLoss
            weighted      = True
            kernel_size   = 7
            lr            = 0.01
            momentum      = 0.9
            
            [Parallelism]
            
            independent_runs = True
            master_port = 5678
            
            [Testing]
            
            num_classes = 32
            '''
        hparams_path = os.path.join(dst_dir, 'config.cfg')
        with open(hparams_path, 'w') as fd:
            fd.write(txt)
            
        return hparams_path

    #------------------------------------
    # read_ints_from_csv
    #-------------------
    
    def read_ints_from_csv(self, csv_path):
        '''
        Read a csv expected to be all integers.
        Return a 2D array with the numbers.
        
        :param csv_path: path to csv file
        :type csv_path: str
        :return: array of contents
        :rtype: [[int]]
        '''
        rows = []
        with open(csv_path,'r') as fd:
            reader = csv.DictReader(fd)
            for row_dict in reader:
                row = list(row_dict.values())
                # Turn string-ints into ints:
                row = [int(el) for el in row]
                rows.append(row)
        return rows

    #------------------------------------
    # read_csv_file
    #-------------------
    
    def read_csv_file(self, key):
        '''
        Returns a dict from <exp_root>/csv_files/<key>.csv
        Used to verify saving/reading
        
        :param key: file name without extension
        :type key: str
        :return content of csv file as dict
        :rtype:  {str : Any}
        '''
        with open(os.path.join(self.exp.root, f'csv_files/{key}.csv')) as fd:
            reader = csv.DictReader(fd)
            row_dicts = list(reader)
            return row_dicts 


    #------------------------------------
    # assert_dataframes_equal
    #-------------------
    
    def assertDataframesEqual(self, df1, df2):
        self.assertTrue((df1.columns == df2.columns).all())
        self.assertTrue((df1.index   == df2.index).all())
        self.assertTrue((df1 == df2).all().all())
        
    #------------------------------------
    # assertPytorchModelsEqual
    #-------------------
    
    def assertPytorchModelsEqual(self, m1, m2):
        
        state1 = m1.state_dict()
        state2 = m2.state_dict()
        
        for m1_val, m2_val in zip(state1.values(), state2.values()):
            self.assertTrue((m1_val == m2_val).all().item())

    #------------------------------------
    # assertScorchModelsEqual
    #-------------------
    
    def assertSkorchModelsEqual(self, m1, m2):
        
        state1 = m1.module_.state_dict()
        state2 = m2.module_.state_dict()
        
        for m1_val, m2_val in zip(state1.values(), state2.values()):
            self.assertTrue((m1_val == m2_val).all().item())



# -------------------- TinyModel --------------
class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(64 * 64, 4096)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.l1(x.view(x.size(0), -1)))

# --------------- Main -------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()