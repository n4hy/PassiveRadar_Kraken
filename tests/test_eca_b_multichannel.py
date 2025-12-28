import sys
# Inject mock before importing the module under test
import mock_gnuradio

import unittest
import numpy as np
import ctypes
from unittest.mock import MagicMock, patch

# Mock ctypes before importing the block to avoid LoadLibrary failing
with patch('ctypes.cdll.LoadLibrary') as mock_load:
    # Setup the mock library return values
    mock_lib = MagicMock()
    mock_load.return_value = mock_lib
    # Set restype/argtypes defaults so assignment doesn't fail
    mock_lib.eca_b_create.restype = ctypes.c_void_p
    mock_lib.eca_b_destroy.restype = None
    mock_lib.eca_b_process.restype = None

    from kraken_passive_radar.eca_b_clutter_canceller import EcaBClutterCanceller

class TestEcaBMultiChannel(unittest.TestCase):
    def setUp(self):
        # We need to patch LoadLibrary again because it's called inside __init__
        self.patcher = patch('ctypes.cdll.LoadLibrary')
        self.mock_load = self.patcher.start()

        # Configure the mock returned by LoadLibrary
        self.mock_lib = MagicMock()
        self.mock_load.return_value = self.mock_lib

        # Ensure we return valid (non-NULL) pointers for create
        self.mock_lib.eca_b_create.return_value = ctypes.c_void_p(12345)

    def tearDown(self):
        self.patcher.stop()

    def test_initialization(self):
        """Test that the block initializes with correct signature for multiple channels."""
        # 4 surveillance channels
        blk = EcaBClutterCanceller(num_taps=64, num_surv_channels=4)
        self.assertEqual(blk.num_surv_channels, 4)

        # Check internal states were created
        self.assertEqual(len(blk._states), 4)
        # Verify create was called 4 times
        self.assertEqual(self.mock_lib.eca_b_create.call_count, 4)

    def test_processing_loop(self):
        """Test that general_work processes all channels."""
        blk = EcaBClutterCanceller(num_taps=64, num_surv_channels=2)

        # Create dummy inputs
        n = 100
        ref = np.zeros(n, dtype=np.complex64)
        surv1 = np.zeros(n, dtype=np.complex64)
        surv2 = np.zeros(n, dtype=np.complex64)

        out1 = np.zeros(n, dtype=np.complex64)
        out2 = np.zeros(n, dtype=np.complex64)

        input_items = [ref, surv1, surv2]
        output_items = [out1, out2]

        # Mock consume
        blk.consume = lambda port, N: None

        # Run work
        processed = blk.general_work(input_items, output_items)

        self.assertEqual(processed, n)
        # Verify process was called 2 times (once per surveillance channel)
        self.assertEqual(self.mock_lib.eca_b_process.call_count, 2)

if __name__ == '__main__':
    unittest.main()
