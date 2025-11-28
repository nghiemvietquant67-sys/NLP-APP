import io
import sys
import unittest
from subprocess import run, PIPE
import os


class TestLab6Examples(unittest.TestCase):
    def test_run_examples_script(self):
        # Ensure the example script runs (it may download models; this test only checks script executes)
        script = os.path.join('Lab_6', 'examples', 'run_lab6_examples.py')
        if not os.path.exists(script):
            self.skipTest('run_lab6_examples.py not found')
        completed = run([sys.executable, script], stdout=PIPE, stderr=PIPE, timeout=300)
        self.assertEqual(completed.returncode, 0, msg=f"Script failed: {completed.stderr.decode()}")
        out = completed.stdout.decode()
        self.assertIn('Fill-Mask results', out)


if __name__ == '__main__':
    unittest.main()
