import unittest

from test import base


class TestVersions(base.TestBase, unittest.TestCase):

    def test_clean_environment(self):
        self.call_main('versions')
        self.assertEqual(self.get_stdout(), '''Available versions:
  v2
  v3
  v4
  v5
  v5-cuda8
  v51
  v51-cuda8

Installed versions:
''')
