
import sys
import unittest
from unittest.mock import MagicMock, patch

# Import the mock environment first
import mock_gnuradio

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../gr-kraken_passive_radar/python')))

from kraken_passive_radar.krakensdr_source import krakensdr_source

class TestKrakenSDRSource(unittest.TestCase):
    def test_initialization(self):
        """Test that the block initializes osmosdr.source with correct arguments"""
        mock_gnuradio.osmosdr.source.reset_mock()
        mock_gnuradio.osmosdr.source.side_effect = None # Clear any side effects

        freq = 100e6
        rate = 2.4e6
        gain = 40.0

        blk = krakensdr_source(frequency=freq, sample_rate=rate, gain=gain)

        # Generate expected args using same logic as class to be robust
        channel_args = []
        for i in range(5):
            channel_args.append(f"rtl={1000+i},buffers=128,buflen=65536")
        expected_args = "numchan=5 " + " ".join(channel_args)

        mock_gnuradio.osmosdr.source.assert_called_with(args=expected_args)

        osmo_instance = mock_gnuradio.osmosdr.source.return_value

        osmo_instance.set_sample_rate.assert_called_with(rate)
        self.assertEqual(osmo_instance.set_center_freq.call_count, 5)
        self.assertEqual(osmo_instance.set_gain.call_count, 5)

    def test_setters(self):
        """Test the setter methods"""
        mock_gnuradio.osmosdr.source.reset_mock()
        mock_gnuradio.osmosdr.source.return_value.reset_mock()
        mock_gnuradio.osmosdr.source.side_effect = None

        blk = krakensdr_source()
        osmo_instance = mock_gnuradio.osmosdr.source.return_value

        # Reset mock counts after initialization to test setters cleanly
        osmo_instance.reset_mock()

        blk.set_frequency(90e6)
        self.assertEqual(osmo_instance.set_center_freq.call_count, 5)

        osmo_instance.reset_mock()
        blk.set_sample_rate(2.0e6)
        osmo_instance.set_sample_rate.assert_called_with(2.0e6)

        osmo_instance.reset_mock()
        blk.set_gain(10)
        self.assertEqual(osmo_instance.set_gain.call_count, 5)

if __name__ == "__main__":
    unittest.main()
