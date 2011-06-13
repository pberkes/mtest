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
        os.makedirs(os.path.join(self.tmpdir, 'mldata'))

    def tearDown(self):
        # remove temporary dir
        shutil.rmtree(self.tmpdir)

    def test_mtest_identical(self):
        for _ in range(3):
            x = np.random.randn(3)
            y = x.copy()
            data_value, pval, N = mtest.mtest(x, y)
            assert pval > 0.1

    def test_mtest_verydifferent(self):
        for _ in range(3):
            x = np.random.normal(loc=0., scale=1., size=(5,))
            y = np.random.normal(loc=5., scale=1., size=(5,))
            data_value, pval, N = mtest.mtest(x, y)
            assert pval < 0.01

if __name__ == '__main__':
    unittest.main()
