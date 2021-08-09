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

import torch

from experiment_manager.experiment_manager import ExperimentManager, AutoSaveThread
from experiment_manager.neural_net_config import NeuralNetConfig
import numpy as np
import pandas as pd


#**********TEST_ALL = True
TEST_ALL = False

'''
TODO:
   o test moving the experiment: ensure relative addressing!
'''

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
        tiny_model = TinyModel()
        torch.save(tiny_model, model_path)
        
        # Create two little csv files:
        csvs_dir   = os.path.join(self.prefab_exp_root,'csv_files')
        os.makedirs(csvs_dir)
        self.make_csv_files(csvs_dir)
        
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
        
        exp1 = ExperimentManager.load(self.exp_root)
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
        csv_file_path = exp.save(tst_dict, 'first_dict')
        
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
        exp.save(row2_dict, 'first_dict')
        
        # Second row should be [100, 200]:
        with open(csv_file_path, 'r') as fd:
            reader = csv.DictReader(fd)
            row_dict0 = next(reader)
            self.assertEqual(list(row_dict0.values()), ['10','20'])
            row_dict1 = next(reader)
            self.assertEqual(list(row_dict1.values()), ['100','200'])

        # Should be able to just write a row, not a dict:
        exp.save([1000,2000], 'first_dict')
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
        csv_file_path = exp.save(tst_dict, 'first_dict')
        exp.close()
        
        del exp
        
        # Reconstitute the same experiment:
        exp = ExperimentManager.load(self.exp_root)
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
    # test_saving_hparams
    #-------------------
    
    #******@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_saving_hparams(self):
        
        exp = ExperimentManager(self.exp_root)

        exp.add_hparams(self.hparams_path, 'my_config')
        config_obj = exp['my_config']
                         
        # Should have a json export of the config instance:
        saved_copy_path = os.path.join(exp.hparams_path, 'my_config.json')
        with open(saved_copy_path, 'r') as fd:
            json_str = fd.read()
            other_config_obj = NeuralNetConfig.from_json(json_str)
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
        exp1 = ExperimentManager.load(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp1
        
        config_obj = exp1['my_config']
        self.assertEqual(config_obj['Training']['net_name'], 'resnet18')
        self.assertEqual(config_obj.getint('Parallelism', 'master_port'), 5678)

    #------------------------------------
    # test_saving_dataframes
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_saving_dataframes(self):
        
        exp = ExperimentManager(self.exp_root)

        df = pd.DataFrame([[1,2,3],
                           [4,5,6],
                           [7,8,9]], 
                           columns=['foo','bar','fum'], 
                           index= ['row1','row2','row3'])

        # Save without the row labels (i.e w/o the index):
        dst_without_idx = exp.save(df, 'mydf')
        self.assertEqual(dst_without_idx, exp.csv_writers['mydf'].fd.name)

        df_retrieved_no_idx_saved = pd.read_csv(dst_without_idx)
        # Should have:
        #        foo  bar  fum
        #    0    1    2    3
        #    1    4    5    6
        #    2    7    8    9
        
        df_true_no_idx = pd.DataFrame.from_dict({'foo' : [1,4,7], 'bar' : [2,5,8], 'fum' : [3,6,9]})
        self.assertTrue((df_retrieved_no_idx_saved == df_true_no_idx).all().all())

        # Now save with index:
        dst_with_idx = exp.save(df, 'mydf_with_idx', index_col='My Col Labels')
        df_true_with_idx = df_true_no_idx.copy()
        df_true_with_idx.index = ['row1', 'row2', 'row3']
        df_retrieved_with_idx_saved = pd.read_csv(dst_with_idx, index_col='My Col Labels')
        # Should have:
        #           foo  bar  fum
        #    row1    1    2    3
        #    row2    4    5    6
        #    row3    7    8    9
        self.assertTrue((df_retrieved_with_idx_saved == df_true_with_idx).all().all())
        exp.close()
        
        del exp
        
        # Reconstitute the same experiment:
        exp1 = ExperimentManager.load(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp1
        
        # All the above tests should work again:
        
        df_no_idx   = pd.read_csv(exp1.abspath('mydf', 'csv'))
        df_with_idx = pd.read_csv(exp1.abspath('mydf_with_idx', 'csv'), 
                                  index_col='My Col Labels')
        self.assertTrue((df_no_idx == df_true_no_idx).all().all())
        self.assertTrue((df_with_idx == df_true_with_idx).all().all())

    #------------------------------------
    # test_saving_series
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_saving_series(self):
        
        exp = ExperimentManager(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp
        my_series = pd.Series([1,2,3], index=['One', 'Two', 'Three'])
        
        dst = exp.save(my_series, 'series_test')
        # Get a dataframe whose first row is the series:
        series_read = pd.read_csv(dst)
        first_row   = series_read.iloc[0,:] 
        self.assertTrue((first_row == my_series).all())

        exp.close()
        
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
        exp.save(my_dict, 'my_results')
        
        # Add a data frame to the same csv:
        df = pd.DataFrame([[3,4],[5,6],[7,8]], columns=['foo', 'bar'])
        csv_path = exp.save(df, 'my_results')
        
        rows = self.read_ints_from_csv(csv_path)

        expected = [[1,2],
                    [3,4],
                    [5,6],
                    [7,8]
                    ]
        self.assertListEqual(rows, expected)
        
        # Add a row in the form of a pd Series:
        ser = pd.Series([9,10], index=['foo', 'bar'])
        exp.save(ser, 'my_results')
        expected.append([9,10])
        rows = self.read_ints_from_csv(csv_path)
        self.assertListEqual(rows, expected)
        
        # Add an np array:
        nparr = np.array([[11,12],[13,14]])
        exp.save(nparr, 'my_results')
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
        exp1 = ExperimentManager.load(self.exp_root)
        
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
        
        exp = ExperimentManager.load(self.exp_root)
        self.assertDictEqual(exp['my_dict'], animal_dict)
        
        exp.auto_save_thread.cancel()

    #------------------------------------
    # test_root_movability
    #-------------------
    
    #******@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_root_movability(self):
        exp = ExperimentManager(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp
        exp['foo'] = 10
        index_strs = ['row0', 'row1']
        col_strs   = ['foo', 'bar', 'fum']
        tst_df = pd.DataFrame([[1,2,3], [4,5,6]], columns=col_strs, index=index_strs)
        exp.save(tst_df, 'my_df')
        config = NeuralNetConfig(self.hparams_path)
        exp.save(config, 'my_config')

        exp.close()
        del exp
        
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='exp_man_tests') as tmp_dir_name:
            shutil.move(self.exp_root, tmp_dir_name)
            new_root = os.path.join(tmp_dir_name, Path(self.exp_root).stem)
            exp1 = ExperimentManager(new_root)
            
            self.assertEqual(exp1['foo'], 10)
            
            df_path = exp1.abspath('my_df', 'csv')
            recovered_df = pd.read_csv(df_path)
            print(recovered_df)
            
            recovered_config = exp1['my_config']
            self.assertDictEqual(recovered_config, config)

            exp1.close()
            shutil.move(new_root, Path(self.exp_root).parent)
                        
        print('foo')

    #------------------------------------
    # test_abspath
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_abspath(self):
        exp = ExperimentManager(self.exp_root)
        # For cleanup in tearDown():
        self.exp = exp

        tst_dict = {'foo' : 10, 'bar' : 20}
        csv_file_path = exp.save(tst_dict, 'first_dict')

        self.assertEqual(exp.abspath('first_dict', 'csv'), csv_file_path)

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
        fld_names = exp._get_field_names(item, index_col='DfIndex')
        self.assertEqual(fld_names, ['DfIndex', 'foo', 'bar', 'fum'])
        
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