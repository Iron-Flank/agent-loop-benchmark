import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from almm_adapter.conformance import check_adapter, sample_manifest
from almm_adapter.http import HttpAdapter


class ProcessAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = subprocess.Popen(
            [sys.executable, '-m', 'almm_adapter', 'serve', '--port', '0'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        line = cls.process.stdout.readline()
        if not line:
            _, error = cls.process.communicate(timeout=5)
            raise AssertionError(error)
        cls.url = json.loads(line)['url']

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.communicate(timeout=5)

    def test_remote_conformance(self):
        self.assertEqual(check_adapter(HttpAdapter(self.url))[-1], 'state reset')

    def test_remote_missing_version_rejected(self):
        manifest = sample_manifest()
        del manifest['adapter']['contractVersion']
        with self.assertRaisesRegex(ValueError, 'contractVersion'):
            HttpAdapter(self.url).initialize(manifest)

    def test_unknown_rpc_not_dispatched(self):
        request = Request(self.url + '/v1/__dict__', data=b'{}',
                          headers={'Content-Type': 'application/json'})
        with self.assertRaises(HTTPError) as result:
            urlopen(request, timeout=5)
        self.assertEqual(result.exception.code, 404)
        result.exception.close()

    def test_gold_data_rejected_remotely(self):
        adapter = HttpAdapter(self.url)
        adapter.initialize(sample_manifest())
        with self.assertRaisesRegex(ValueError, 'expected'):
            adapter.answerProbe({'probeId': 'p', 'question': '?', 'expected': {}})

    def test_invalid_json_rejected(self):
        request = Request(self.url + '/v1/initialize', data=b'not json',
                          headers={'Content-Type': 'application/json'})
        with self.assertRaises(HTTPError) as result:
            urlopen(request, timeout=5)
        self.assertEqual(result.exception.code, 400)
        result.exception.close()

    def test_copied_template_fails_with_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'new-adapter'
            shutil.copytree('templates/runtime-adapter', target)
            environment = dict(os.environ, PYTHONPATH=str(Path.cwd()))
            result = subprocess.run([sys.executable, 'conformance.py'], cwd=target,
                                    env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('Implement initialize(runManifest)', result.stderr)


if __name__ == '__main__':
    unittest.main()
