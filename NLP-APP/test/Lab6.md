import os
import unittest


class TestReadme(unittest.TestCase):
    def test_readme_exists(self):
        self.assertTrue(os.path.exists('README.md'))


if __name__ == '__main__':
    unittest.main()
