import unittest

from test import base


class TestVersion(base.TestBase, unittest.TestCase):

    def test_clean_environment(self):
        self.call_main('version')
        self.assertEqual(self.get_stdout(), '(none)\n')
