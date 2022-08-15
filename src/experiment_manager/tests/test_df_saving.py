'''
Created on Aug 6, 2022

@author: paepcke
'''
import os
import shutil
import unittest

from experiment_manager.experiment_manager import ExperimentManager, Datatype

import pandas as pd
import numpy as np

TEST_ALL = True
#TEST_ALL = False


class IdxType:
    simple = pd.Index(pd.RangeIndex(4))
    named  = pd.Index(['row1', 'row2', 'row3', 'row4'])
    multi  = pd.MultiIndex.from_tuples([('blue', 10), ('blue', 20), ('red', 30), ('red', 40)])

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
        
        df = self.make_simple_df(IdxType.simple)
        self.exp.save('simple_idx', df)
        df_retrieved = self.exp.read('simple_idx', Datatype.tabular)
        expected = \
            ('   Col1  Col2  Col3\n'
            '0     1     2     3\n'
            '1     4     5     6\n'
            '2     7     8     9\n'
            '3    10    11    12')

        self.assertTrue(self.cmp_df_str(df_retrieved, expected))
        self.assertTrue((df == df_retrieved).all().all())

    #------------------------------------
    # testSimpleIdxNoIdxColWithIdxName
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testSimpleIdxNoIdxColWithIdxName(self):
        
        df = self.make_simple_df(IdxType.simple)
        df.index.name = 'my_index'
        self.exp.save('simple_idx', df)
        df_retrieved = self.exp.read('simple_idx', Datatype.tabular)
        expected = \
            ('          Col1  Col2  Col3\n'
             'my_index                  \n'
             '0            1     2     3\n'
             '1            4     5     6\n'
             '2            7     8     9\n'
             '3           10    11    12')        
         
        self.assertTrue(self.cmp_df_str(df_retrieved, expected))
        self.assertTrue((df == df_retrieved).all().all())

    #------------------------------------
    # test_multi_index_no_index_col
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_multi_index_no_index_col(self):
        df = self.make_simple_df(IdxType.multi)
        self.exp.save('multi_idx', df)
        df_retrieved = self.exp.read('multi_idx', Datatype.tabular)

        expected = \
            ('         Col1  Col2  Col3\n'
            'blue 10     1     2     3\n'
            '     20     4     5     6\n'
            'red  30     7     8     9\n'
            '     40    10    11    12')

        self.assertTrue(self.cmp_df_str(df_retrieved, expected))
        self.assertTrue((df == df_retrieved).all().all())

    #------------------------------------
    # test_multi_index_with_index_col_name
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_multi_index_with_index_col_name(self):
        df = self.make_simple_df(IdxType.multi)
        df.index.set_names(['color', 'name'], level=[0,1], inplace=True)
        self.exp.save('multi_idx', df)
        df_retrieved = self.exp.read('multi_idx', Datatype.tabular)

        expected = \
            ('            Col1  Col2  Col3\n'
            'color name                  \n'
            'blue  10       1     2     3\n'
            '      20       4     5     6\n'
            'red   30       7     8     9\n'
            '      40      10    11    12')
        
        self.assertTrue(self.cmp_df_str(df_retrieved, expected))
        self.assertTrue((df == df_retrieved).all().all())

    #------------------------------------
    # test_multi_idxs_both_no_names
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_multi_idxs_both_no_names(self):
        df = self.make_df_row_and_col_idxs(row_idx_nm=None, col_idx_nm=None)
        self.exp.save('dbl_multi_idx', df)
        df_retrieved = self.exp.read('dbl_multi_idx', Datatype.tabular)
        expected = \
            ('                     lev1_a                             lev1_b2       \n'
            '                     lev2_a        lev2_b                lev2_a lev2_b\n'
            '                     lev3_a lev3_b lev3_a lev3_b lev3_c  lev3_a lev3_b\n'
            'row1_a row2_a row3_a      0      1      2      3      4       5      6\n'
            '              row3_b      7      8      9     10     11      12     13\n'
            '       row2_b row3_a     14     15     16     17     18      19     20\n'
            '              row3_b     21     22     23     24     25      26     27\n'
            '              row3_c     28     29     30     31     32      33     34\n'
            'row1_b row2_a row3_a     35     36     37     38     39      40     41\n'
            '       row2_b row3_a     42     43     44     45     46      47     48')
        
        self.assertTrue(self.cmp_df_str(df_retrieved, expected))
        self.assertTrue((df == df_retrieved).all().all())

    #------------------------------------
    # test_multi_idxs_rows_with_names
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_multi_idxs_rows_with_names(self):
        df = self.make_df_row_and_col_idxs(row_idx_nm=['Rlevel0', 'Rlevel1', 'Rlevel2'], 
                                           col_idx_nm=None)
        self.exp.save('dbl_multi_idx', df)
        df_retrieved = self.exp.read('dbl_multi_idx', Datatype.tabular)
        expected = \
            ('                        lev1_a                             lev1_b2       \n'
            '                        lev2_a        lev2_b                lev2_a lev2_b\n'
            '                        lev3_a lev3_b lev3_a lev3_b lev3_c  lev3_a lev3_b\n'
            'Rlevel0 Rlevel1 Rlevel2                                                  \n'
            'row1_a  row2_a  row3_a       0      1      2      3      4       5      6\n'
            '                row3_b       7      8      9     10     11      12     13\n'
            '        row2_b  row3_a      14     15     16     17     18      19     20\n'
            '                row3_b      21     22     23     24     25      26     27\n'
            '                row3_c      28     29     30     31     32      33     34\n'
            'row1_b  row2_a  row3_a      35     36     37     38     39      40     41\n'
            '        row2_b  row3_a      42     43     44     45     46      47     48')
        
        self.assertTrue(self.cmp_df_str(df_retrieved, expected))
        self.assertTrue((df == df_retrieved).all().all())
        self.assertListEqual(df.index.names, ['Rlevel0', 'Rlevel1', 'Rlevel2'])
        self.assertListEqual(df.columns.names, [None, None, None])

    #------------------------------------
    # test_multi_idxs_cols_with_names
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_multi_idxs_cols_with_names(self):
        df = self.make_df_row_and_col_idxs(row_idx_nm=None,
                                           col_idx_nm=['Clevel0', 'Clevel1', 'Clevel2'])
        self.exp.save('dbl_multi_idx', df)
        df_retrieved = self.exp.read('dbl_multi_idx', Datatype.tabular)
        expected = \
            ('Clevel0              lev1_a                             lev1_b2       \n'
             'Clevel1              lev2_a        lev2_b                lev2_a lev2_b\n'
             'Clevel2              lev3_a lev3_b lev3_a lev3_b lev3_c  lev3_a lev3_b\n'
             'row1_a row2_a row3_a      0      1      2      3      4       5      6\n'
             '              row3_b      7      8      9     10     11      12     13\n'
             '       row2_b row3_a     14     15     16     17     18      19     20\n'
             '              row3_b     21     22     23     24     25      26     27\n'
             '              row3_c     28     29     30     31     32      33     34\n'
             'row1_b row2_a row3_a     35     36     37     38     39      40     41\n'
             '       row2_b row3_a     42     43     44     45     46      47     48')

        self.assertTrue(self.cmp_df_str(df_retrieved, expected))
        self.assertTrue((df == df_retrieved).all().all())
        self.assertListEqual(df.index.names, [None, None, None])
        self.assertListEqual(df.columns.names, ['Clevel0', 'Clevel1', 'Clevel2'])

    #------------------------------------
    # test_multi_idxs_both_with_names
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_multi_idxs_both_with_names(self):
        df = self.make_df_row_and_col_idxs(row_idx_nm=['Rlevel0', 'Rlevel1', 'Rlevel2'], 
                                           col_idx_nm=['Clevel0', 'Clevel1', 'Clevel2'])
        self.exp.save('dbl_multi_idx', df)
        df_retrieved = self.exp.read('dbl_multi_idx', Datatype.tabular)
        expected = \
            ('Clevel0                 lev1_a                             lev1_b2       \n'
            'Clevel1                 lev2_a        lev2_b                lev2_a lev2_b\n'
            'Clevel2                 lev3_a lev3_b lev3_a lev3_b lev3_c  lev3_a lev3_b\n'
            'Rlevel0 Rlevel1 Rlevel2                                                  \n'
            'row1_a  row2_a  row3_a       0      1      2      3      4       5      6\n'
            '                row3_b       7      8      9     10     11      12     13\n'
            '        row2_b  row3_a      14     15     16     17     18      19     20\n'
            '                row3_b      21     22     23     24     25      26     27\n'
            '                row3_c      28     29     30     31     32      33     34\n'
            'row1_b  row2_a  row3_a      35     36     37     38     39      40     41\n'
            '        row2_b  row3_a      42     43     44     45     46      47     48')

        self.assertTrue(self.cmp_df_str(df_retrieved, expected))
        self.assertTrue((df == df_retrieved).all().all())
        self.assertListEqual(df.index.names, ['Rlevel0', 'Rlevel1', 'Rlevel2'])
        self.assertListEqual(df.columns.names, ['Clevel0', 'Clevel1', 'Clevel2'])

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
    # make_simple_df
    #-------------------
    

    def make_simple_df(self, idx_type, idx_col_nm=None):
        
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
    # make_df_row_and_col_idxs
    #-------------------
    
    def make_df_row_and_col_idxs(self, row_idx_nm=None, col_idx_nm=None):
        '''
        Create and return:
                              lev1_a                             lev1_b2       
                              lev2_a        lev2_b                lev2_a lev2_b
                              lev3_a lev3_b lev3_a lev3_b lev3_c  lev3_a lev3_b
         row1_a row2_a row3_a      0      1      2      3      4       5      6
                       row3_b      7      8      9     10     11      12     13
                row2_b row3_a     14     15     16     17     18      19     20
                       row3_b     21     22     23     24     25      26     27
                       row3_c     28     29     30     31     32      33     34
         row1_b row2_a row3_a     35     36     37     38     39      40     41
                row2_b row3_a     42     43     44     45     46      47     48
        '''
        
        col_idx_arrs = [['lev1_a', 'lev1_a', 'lev1_a', 'lev1_a', 'lev1_a', 'lev1_b2', 'lev1_b2'],
                        ['lev2_a', 'lev2_a', 'lev2_b', 'lev2_b', 'lev2_b', 'lev2_a', 'lev2_b'],
                        ['lev3_a', 'lev3_b', 'lev3_a', 'lev3_b', 'lev3_c', 'lev3_a', 'lev3_b']
                        ]
        multi_idx_cols = pd.MultiIndex.from_arrays(col_idx_arrs)
        
        row_idx_arrs = [['row1_a', 'row1_a', 'row1_a', 'row1_a', 'row1_a', 'row1_b', 'row1_b'],
                        ['row2_a', 'row2_a', 'row2_b', 'row2_b', 'row2_b', 'row2_a', 'row2_b'],
                        ['row3_a', 'row3_b', 'row3_a', 'row3_b', 'row3_c', 'row3_a', 'row3_a']
                        ]
        multi_idx_rows = pd.MultiIndex.from_arrays(row_idx_arrs)
        
        df_both = pd.DataFrame(np.zeros((len(multi_idx_rows), 
                                         len(multi_idx_cols))), 
                                         columns=multi_idx_cols, 
                                         index=multi_idx_rows, 
                                         dtype=int)
        content = np.arange(49).reshape((7,7))
        df_both.loc[:,:] = content
        
        if row_idx_nm is not None:
            df_both.index.names = row_idx_nm
            
        if col_idx_nm is not None:
            df_both.columns.names = col_idx_nm
        
        return df_both

    
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