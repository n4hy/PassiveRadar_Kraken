import sys
import os
import unittest
import importlib
from unittest.mock import MagicMock, patch

# Import the mock environment first
import mock_gnuradio

# OOT module path must come BEFORE display package to resolve namespace correctly
oot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../gr-kraken_passive_radar/python'))
sys.path.insert(0, oot_path)

# Force fresh import of krakensdr_source to pick up our mocks
# (other test files may have imported with different mock state)
for mod_name in list(sys.modules.keys()):
    if 'kraken_passive_radar' in mod_name:
        del sys.modules[mod_name]

from kraken_passive_radar.krakensdr_source import krakensdr_source
_ks_module = sys.modules['kraken_passive_radar.krakensdr_source']


class TestKrakenSDRSource(unittest.TestCase):
    def test_initialization(self):
        """Test that the block initializes osmosdr.source with correct arguments"""
        mock_osmosdr = MagicMock()
        with patch.object(_ks_module, 'osmosdr', mock_osmosdr):
            freq = 100e6
            rate = 2.4e6
            gain = 40.0

            blk = krakensdr_source(frequency=freq, sample_rate=rate, gain=gain)

            # Generate expected args using same logic as class to be robust
            channel_args = []
            for i in range(5):
                channel_args.append(f"rtl={1000+i},buffers=128,buflen=65536")
            expected_args = "numchan=5 " + " ".join(channel_args)

            mock_osmosdr.source.assert_called_with(args=expected_args)

            osmo_instance = mock_osmosdr.source.return_value
            osmo_instance.set_sample_rate.assert_called_with(rate)
            self.assertEqual(osmo_instance.set_center_freq.call_count, 5)
            self.assertEqual(osmo_instance.set_gain.call_count, 5)

    def test_setters(self):
        """Test the setter methods"""
        mock_osmosdr = MagicMock()
        with patch.object(_ks_module, 'osmosdr', mock_osmosdr):
            blk = krakensdr_source()
            osmo_instance = mock_osmosdr.source.return_value

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
