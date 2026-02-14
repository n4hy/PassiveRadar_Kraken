import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Import the mock environment first
import mock_gnuradio

# OOT module path
oot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../gr-kraken_passive_radar/python'))
sys.path.insert(0, oot_path)

# Force fresh import to pick up our mocks
for mod_name in list(sys.modules.keys()):
    if 'kraken_passive_radar' in mod_name:
        del sys.modules[mod_name]

from kraken_passive_radar.rspdx_source import rspdx_source
_rs_module = sys.modules['kraken_passive_radar.rspdx_source']


class TestRSPdxSource(unittest.TestCase):

    def test_initialization_defaults(self):
        """Test block initializes soapy.source with correct default args."""
        mock_soapy = MagicMock()
        with patch.object(_rs_module, 'soapy', mock_soapy):
            blk = rspdx_source()

            mock_soapy.source.assert_called_once_with(
                'driver=sdrplay', 'fc32', 1, '', '', [''], ['']
            )

            src = mock_soapy.source.return_value
            src.set_sample_rate.assert_called_with(0, 2.4e6)
            src.set_frequency.assert_called_with(0, 100e6)
            src.set_antenna.assert_called_with(0, 'Antenna A')
            src.set_gain.assert_any_call(0, 'IFGR', 40.0)
            src.set_gain.assert_any_call(0, 'RFGR', 0.0)

            # Bandwidth 0 = auto, so set_bandwidth should NOT be called
            src.set_bandwidth.assert_not_called()

            # Device settings applied
            src.write_setting.assert_any_call('biasT_ctrl', 'false')
            src.write_setting.assert_any_call('rfnotch_ctrl', 'true')
            src.write_setting.assert_any_call('dabnotch_ctrl', 'true')
            src.write_setting.assert_any_call('hdr_ctrl', 'false')

    def test_initialization_custom(self):
        """Test block initializes with custom parameters."""
        mock_soapy = MagicMock()
        with patch.object(_rs_module, 'soapy', mock_soapy):
            blk = rspdx_source(
                frequency=433e6,
                sample_rate=6e6,
                if_gain=50.0,
                rf_gain=10.0,
                antenna='Antenna B',
                bandwidth=5e6,
                bias_t=True,
                rf_notch=False,
                dab_notch=False,
                hdr_mode=True
            )

            src = mock_soapy.source.return_value
            src.set_frequency.assert_called_with(0, 433e6)
            src.set_sample_rate.assert_called_with(0, 6e6)
            src.set_gain.assert_any_call(0, 'IFGR', 50.0)
            src.set_gain.assert_any_call(0, 'RFGR', 10.0)
            src.set_antenna.assert_called_with(0, 'Antenna B')
            src.set_bandwidth.assert_called_with(0, 5e6)

            src.write_setting.assert_any_call('biasT_ctrl', 'true')
            src.write_setting.assert_any_call('rfnotch_ctrl', 'false')
            src.write_setting.assert_any_call('dabnotch_ctrl', 'false')
            src.write_setting.assert_any_call('hdr_ctrl', 'true')

    def test_set_frequency(self):
        """Test runtime frequency change."""
        mock_soapy = MagicMock()
        with patch.object(_rs_module, 'soapy', mock_soapy):
            blk = rspdx_source()
            src = mock_soapy.source.return_value
            src.reset_mock()

            blk.set_frequency(915e6)
            src.set_frequency.assert_called_once_with(0, 915e6)
            self.assertEqual(blk.frequency, 915e6)

    def test_set_sample_rate(self):
        """Test runtime sample rate change."""
        mock_soapy = MagicMock()
        with patch.object(_rs_module, 'soapy', mock_soapy):
            blk = rspdx_source()
            src = mock_soapy.source.return_value
            src.reset_mock()

            blk.set_sample_rate(8e6)
            src.set_sample_rate.assert_called_once_with(0, 8e6)
            self.assertEqual(blk.sample_rate, 8e6)

    def test_set_if_gain(self):
        """Test runtime IF gain change."""
        mock_soapy = MagicMock()
        with patch.object(_rs_module, 'soapy', mock_soapy):
            blk = rspdx_source()
            src = mock_soapy.source.return_value
            src.reset_mock()

            blk.set_if_gain(55.0)
            src.set_gain.assert_called_once_with(0, 'IFGR', 55.0)
            self.assertEqual(blk.if_gain, 55.0)

    def test_set_rf_gain(self):
        """Test runtime RF gain reduction change."""
        mock_soapy = MagicMock()
        with patch.object(_rs_module, 'soapy', mock_soapy):
            blk = rspdx_source()
            src = mock_soapy.source.return_value
            src.reset_mock()

            blk.set_rf_gain(20.0)
            src.set_gain.assert_called_once_with(0, 'RFGR', 20.0)
            self.assertEqual(blk.rf_gain, 20.0)

    def test_set_antenna(self):
        """Test runtime antenna port change."""
        mock_soapy = MagicMock()
        with patch.object(_rs_module, 'soapy', mock_soapy):
            blk = rspdx_source()
            src = mock_soapy.source.return_value
            src.reset_mock()

            blk.set_antenna('Antenna C')
            src.set_antenna.assert_called_once_with(0, 'Antenna C')
            self.assertEqual(blk.antenna, 'Antenna C')

    def test_set_bandwidth(self):
        """Test runtime bandwidth change."""
        mock_soapy = MagicMock()
        with patch.object(_rs_module, 'soapy', mock_soapy):
            blk = rspdx_source()
            src = mock_soapy.source.return_value
            src.reset_mock()

            blk.set_bandwidth(1.536e6)
            src.set_bandwidth.assert_called_once_with(0, 1.536e6)

    def test_set_bias_t(self):
        """Test runtime Bias-T toggle."""
        mock_soapy = MagicMock()
        with patch.object(_rs_module, 'soapy', mock_soapy):
            blk = rspdx_source()
            src = mock_soapy.source.return_value
            src.reset_mock()

            blk.set_bias_t(True)
            src.write_setting.assert_called_with('biasT_ctrl', 'true')

    def test_set_hdr_mode(self):
        """Test runtime HDR mode toggle."""
        mock_soapy = MagicMock()
        with patch.object(_rs_module, 'soapy', mock_soapy):
            blk = rspdx_source()
            src = mock_soapy.source.return_value
            src.reset_mock()

            blk.set_hdr_mode(True)
            src.write_setting.assert_called_with('hdr_ctrl', 'true')

    def test_device_settings_error_handling(self):
        """Test that device setting failures don't crash initialization."""
        mock_soapy = MagicMock()
        mock_soapy.source.return_value.write_setting.side_effect = RuntimeError("not supported")
        with patch.object(_rs_module, 'soapy', mock_soapy):
            # Should not raise despite write_setting errors
            blk = rspdx_source(bias_t=True, hdr_mode=True)
            self.assertIsNotNone(blk)


if __name__ == "__main__":
    unittest.main()
