# Author: Pietro Berkes < pietro _DOT_ berkes _AT_ googlemail _DOT_ com >
# Copyright (c) 2011 Pietro Berkes
# License: GPL v3

import tempfile
import shutil
import os
import numpy as np

import unittest
import mtest

_random_seed = np.random.randint(2**31-1)
print "Execute tests with random seed:", _random_seed

class TestMTest(unittest.TestCase):

    def setUp(self):
        # set random seed
        global _random_seed
        np.random.seed(_random_seed)
        # create temporary dir for the tables
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, 'tables'))

    def tearDown(self):
        # remove temporary dir
        shutil.rmtree(self.tmpdir)

    def test_mtest_build_tables(self):
        x = np.random.randn(3)
        y = x.copy()
        data_value, pval, ncases = mtest.mtest(x, y, min_ncases=20,
                                               path=self.tmpdir)
        assert data_value < 1
        assert pval > 0.1

    def test_mtest_identical(self):
        for _ in range(3):
            x = np.random.randn(3)
            y = x.copy()
            data_value, pval, ncases = mtest.mtest(x, y, min_ncases=1000)
            assert ncases >= 1000
            assert data_value < 1
            assert pval > 0.1

    def test_mtest_verydifferent(self):
        for _ in range(3):
            x = np.random.normal(loc=0., scale=1., size=(5,))
            y = np.random.normal(loc=5., scale=1., size=(5,))
            data_value, pval, ncases = mtest.mtest(x, y, min_ncases=1000)
            assert ncases >= 1000
            assert data_value > 1
            assert pval < 0.01

    def test_thresholds_from_paper(self):
        """Test that one gets the thresholds as in the arxiv paper."""
        thresholds = {3: 19.8, 4: 15.7, 5: 13.7, 10: 10.2, 50: 7.8}
        for n in thresholds:
            res = mtest.typeI_threshold(n, n, 1000)
            # TODO: make this test stronger once mtest ships with larger tables
            assert abs(res - thresholds[n]) < 1.

    def test_typeII_from_paper(self):
        """Test that one gets the type II error as in the arxiv paper."""
        mtypeII_025 = {3: 0.61, 4: 0.40, 5: 0.23, 10: 0., 50: 0.}
        mtypeII_150 = {3: 0.87, 4: 0.83, 5: 0.80, 10: 0.59, 50: 0.}
        ttypeII_025 = {3: 0.68, 4: 0.59, 5: 0.51, 10: 0.19, 50: 0.}
        ttypeII_150 = {3: 0.87, 4: 0.83, 5: 0.80, 10: 0.61, 50: 0.02}

        sgm_2 = 0.25
        for n in mtypeII_025:
            mval, tval = mtest.compare_power(n, n, 1000, 1., sgm_2)
            assert abs(mval - mtypeII_025[n]) < 0.02
            assert abs(tval - ttypeII_025[n]) < 0.02

        print '1.5'
        sgm_2 = 1.5
        for n in mtypeII_150:
            mval, tval = mtest.compare_power(n, n, 1000, 1., sgm_2)
            assert abs(mval - mtypeII_150[n]) < 0.02
            assert abs(tval - ttypeII_150[n]) < 0.02

if __name__ == '__main__':
    unittest.main()
