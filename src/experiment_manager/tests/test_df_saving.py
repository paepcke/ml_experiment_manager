'''
Created on Aug 6, 2022

@author: paepcke
'''
import os
import shutil
import unittest

from experiment_manager.experiment_manager import ExperimentManager, Datatype

import pandas as pd


TEST_ALL = True
#TEST_ALL = False


class IdxType:
    simple = pd.Index(pd.RangeIndex(4))
    named  = pd.Index(['row1', 'row2', 'row3', 'row4'])
    multi  = pd.MultiIndex.from_tuples([('blue', '10'), ('blue', 20), ('red', 30), ('red', 40)])

class ExpManagerDataFrameTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir  = os.path.dirname(__file__)
        cls.tmp_dir  = '/tmp'
        cls.exp_root = os.path.join(cls.tmp_dir, 'exp_tests')

    @classmethod
    def tearDownClass(cls):
        super(ExpManagerDataFrameTester, cls).tearDownClass()
        shutil.rmtree(cls.exp_root)

    def setUp(self):
        self.clear_experiment()

    def tearDown(self):
        pass

# ---------------------  Tests -----------

    #------------------------------------
    # testSimpleIdxNoIdxColNoIdxName
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testSimpleIdxNoIdxColNoIdxName(self):
        
        df = self.make_spectro_df(IdxType.simple)
        self.exp.save('simple_idx', df)
        df_retrieved = self.exp.read('simple_idx', Datatype.tabular)
        expected = \
            ('   Col1  Col2  Col3\n'
            '0     1     2     3\n'
            '1     4     5     6\n'
            '2     7     8     9\n'
            '3    10    11    12')

        self.assertTrue(self.cmp_df_str(df_retrieved, expected))

    #------------------------------------
    # testSimpleIdxNoIdxColWithIdxName
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testSimpleIdxNoIdxColWithIdxName(self):
        
        df = self.make_spectro_df(IdxType.simple)
        df.index.name = 'my_index'
        self.exp.save('simple_idx', df)
        df_retrieved = self.exp.read('simple_idx', Datatype.tabular)
        expected = \
            ('   my_index  Col1  Col2  Col3\n'
            '0         0     1     2     3\n'
            '1         1     4     5     6\n'
            '2         2     7     8     9\n'
            '3         3    10    11    12')
         
        self.assertTrue(self.cmp_df_str(df_retrieved, expected))

        df = self.make_spectro_df(IdxType.multi)
        df.index.rename(('color', 'number'), inplace=True)
        self.exp.save('multi_idx', df)
        df_retrieved = self.exp.read('multi_idx', Datatype.tabular, index_col=('color', 'number'))
        expected = \
            ('              Col1  Col2  Col3\n'
            'color number                  \n'
            'blue  10         1     2     3\n'
            '      20         4     5     6\n'
            'red   30         7     8     9\n'
            '      40        10    11    12')
        
        self.assertTrue(self.cmp_df_str(df_retrieved, expected))

    #------------------------------------
    #  testSimpleIdxWithIdxCol
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testSimpleIdxWithIdxCol(self):
        
        df = self.make_spectro_df(IdxType.simple)
        self.exp.save('simple_idx', df)
        df_retrieved = self.exp.read('simple_idx', Datatype.tabular, index_col='my_idx')
        expected = \
            ('        Col1  Col2  Col3\n'
            'my_idx                  \n'
            '0          1     2     3\n'
            '1          4     5     6\n'
            '2          7     8     9\n'
            '3         10    11    12')
        
        self.assertTrue(self.cmp_df_str(df_retrieved, expected))

    #------------------------------------
    # test_multi_index_no_index_col
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_multi_index_no_index_col(self):
        df = self.make_spectro_df(IdxType.multi)
        self.exp.save('multi_idx', df)
        df_retrieved = self.exp.read('multi_idx', Datatype.tabular)

        expected = \
            ('  Unnamed: 0  Unnamed: 1  Col1  Col2  Col3\n'
            '0       blue          10     1     2     3\n'
            '1       blue          20     4     5     6\n'
            '2        red          30     7     8     9\n'
            '3        red          40    10    11    12')

        self.assertTrue(self.cmp_df_str(df_retrieved, expected))

    #------------------------------------
    # test_multi_index_with_index_col
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_multi_index_with_index_col(self):
        df = self.make_spectro_df(IdxType.multi)
        self.exp.save('multi_idx', df)
        df_retrieved = self.exp.read('multi_idx', Datatype.tabular, index_col=('color', 'name'))

        expected = \
            ('            Col1  Col2  Col3\n'
            'color name                  \n'
            'blue  10       1     2     3\n'
            '      20       4     5     6\n'
            'red   30       7     8     9\n'
            '      40      10    11    12')
        
        self.assertTrue(self.cmp_df_str(df_retrieved, expected))

# ----------------------- Utilities -------------

    #------------------------------------
    # clear_experiment
    #-------------------
    
    def clear_experiment(self):
        try:
            shutil.rmtree(ExpManagerDataFrameTester.exp_root)
        except Exception:
            pass
        self.exp = ExperimentManager(self.exp_root)

    #------------------------------------
    # make_spectro_df
    #-------------------
    
    def make_spectro_df(self, idx_type, idx_col_nm=None):
        
        df = pd.DataFrame([[1,2,3],
                           [4,5,6],
                           [7,8,9],
                           [10,11,12]],
                           columns=['Col1', 'Col2', 'Col3'])
        df.set_index(idx_type, inplace=True)
        if idx_col_nm is not None:
            df.index.rename(idx_col_nm)
        return df
    
    #------------------------------------
    # df_from_str
    #-------------------
    
    def cmp_df_str(self, df, df_str):
        """
        Given a df and string that would print 
        the way a df would print, return true if
        the two args are equivalent:
        
        '''    Unnamed: 0  Col1  Col2  Col3
        0           0     1     2     3
        1           1     4     5     6
        2           2     7     8     9
        3           3    10    11    12'''
        """
        return df.to_string() == df_str


# ----------------------  Main  ----------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()