import unittest

from test import base


class TestDeactivate(base.TestBase, unittest.TestCase):

    def test_clean_environment(self):
        self.call_main('deactivate')
