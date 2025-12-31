
import sys
import unittest
from unittest.mock import MagicMock, patch

# Import the mock environment first
import mock_gnuradio

# Now import the module to test
# We need to make sure the python path is set correctly to find kraken_passive_radar
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../gr-kraken_passive_radar/python')))

from kraken_passive_radar.krakensdr_source import krakensdr_source

class TestKrakenSDRSource(unittest.TestCase):
    def test_initialization(self):
        """Test that the block initializes osmosdr.source with correct arguments"""

        # Instantiate the block
        # The mock_gnuradio module mocks osmosdr, so we can check if it was called

        # Reset the mock to ensure a clean state
        mock_gnuradio.osmosdr.source.reset_mock()

        # Initialize the source block
        freq = 100e6
        rate = 2.4e6
        gain = 40.0

        blk = krakensdr_source(frequency=freq, sample_rate=rate, gain=gain)

        # Verify osmosdr.source was called with numchan=5 and explicit SERIALS
        expected_args = "numchan=5 rtl=1000,buffers=128,buflen=65536 rtl=1001,buffers=128,buflen=65536 rtl=1002,buffers=128,buflen=65536 rtl=1003,buffers=128,buflen=65536 rtl=1004,buffers=128,buflen=65536"
        mock_gnuradio.osmosdr.source.assert_called_with(args=expected_args)

        # Verify settings were applied
        # We need to get the instance returned by osmosdr.source()
        osmo_instance = mock_gnuradio.osmosdr.source.return_value

        osmo_instance.set_sample_rate.assert_called_with(rate)
        # Check center freq was set for all channels
        self.assertEqual(osmo_instance.set_center_freq.call_count, 5)

        # Verify per-channel settings
        # set_gain(gain, channel)
        # We check if it was called 5 times
        self.assertEqual(osmo_instance.set_gain.call_count, 5)
        self.assertEqual(osmo_instance.set_freq_corr.call_count, 5)
        self.assertEqual(osmo_instance.set_dc_offset_mode.call_count, 5)
        self.assertEqual(osmo_instance.set_iq_balance_mode.call_count, 5)

        # Verify connections were made (via self.connect)
        # Since we mocked hier_block2, we can't easily check internal state,
        # but we can check if the connect method of the block (which is mocked) was called.
        # However, in our mock implementation of HierBlock2, connect does nothing and isn't a MagicMock unless we patch it on the instance.
        # But since we are testing logic in __init__, and the code calls self.connect(),
        # if the code runs without error, it means logic is sound regarding the loop.

    def test_setters(self):
        """Test the setter methods"""
        blk = krakensdr_source()
        osmo_instance = mock_gnuradio.osmosdr.source.return_value
        osmo_instance.reset_mock()

        blk.set_frequency(90e6)
        self.assertEqual(osmo_instance.set_center_freq.call_count, 5)

        blk.set_sample_rate(2.0e6)
        osmo_instance.set_sample_rate.assert_called_with(2.0e6)

        blk.set_gain(10)
        # Should be called 5 times
        self.assertEqual(osmo_instance.set_gain.call_count, 5)

if __name__ == "__main__":
    unittest.main()
