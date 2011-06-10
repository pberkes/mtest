# Author: Pietro Berkes < pietro _DOT_ berkes _AT_ googlemail _DOT_ com >
# Copyright (c) 2011 Pietro Berkes
# License: GPL v3

import tempfile
import shutil
import os

import unittest
import .mtest

class TestMTest(unittest.TestCase):
    def setUp(self):
        # create temporary dir for the tables
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, 'mldata'))

    def tearDown(self):
        # remove temporary dir
        shutil.rmtree(self.tmpdir)
        
    def test_mtest_identical(self):
        pass

if __name__ == '__main__':
    unittest.main()
