import sys
import os
import unittest
from unittest.mock import MagicMock, patch, call

# Import the mock environment first
import mock_gnuradio

# OOT module path
oot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../gr-kraken_passive_radar/python'))
sys.path.insert(0, oot_path)

# Force fresh import to pick up our mocks
for mod_name in list(sys.modules.keys()):
    if 'kraken_passive_radar' in mod_name:
        del sys.modules[mod_name]

# Mock sdrplay3 module before importing rspduo_source
mock_sdrplay3 = MagicMock()
mock_sdrplay3.stream_args.return_value = MagicMock(name='stream_args_instance')
sys.modules['gnuradio.sdrplay3'] = mock_sdrplay3
sys.modules['sdrplay3'] = mock_sdrplay3

from kraken_passive_radar.rspduo_source import rspduo_source
_rs_module = sys.modules['kraken_passive_radar.rspduo_source']


class TestRSPduoSource(unittest.TestCase):

    def setUp(self):
        """Reset mock before each test."""
        mock_sdrplay3.reset_mock()
        mock_sdrplay3.stream_args.return_value = MagicMock(name='stream_args_instance')

    def test_initialization_defaults(self):
        """Test block initializes sdrplay3.rspduo with correct default args."""
        with patch.object(_rs_module, 'sdrplay3', mock_sdrplay3):
            blk = rspduo_source()

            # Verify stream_args created correctly
            mock_sdrplay3.stream_args.assert_called_once_with(
                output_type='fc32', channels_size=2
            )

            # Verify rspduo constructor called with correct mode and antenna
            mock_sdrplay3.rspduo.assert_called_once_with(
                '',
                mock_sdrplay3.stream_args.return_value,
                'Dual Tuner (diversity reception)',
                'Both Tuners',
            )

            src = mock_sdrplay3.rspduo.return_value
            src.set_center_freq.assert_called_with(100e6)
            src.set_sample_rate.assert_called_with(2e6)

            # Verify gain negation on both tuners
            src.set_gain.assert_any_call(-40.0, 'IF', 0)
            src.set_gain.assert_any_call(-40.0, 'IF', 1)
            src.set_gain.assert_any_call(-0.0, 'RF', 0)
            src.set_gain.assert_any_call(-0.0, 'RF', 1)

            # Bandwidth 0 = auto, so set_bandwidth should NOT be called
            src.set_bandwidth.assert_not_called()

            # Device settings applied
            src.set_setting.assert_any_call('biasT_ctrl', 'false')
            src.set_setting.assert_any_call('rfnotch_ctrl', 'true')
            src.set_setting.assert_any_call('dabnotch_ctrl', 'true')
            src.set_setting.assert_any_call('amnotch_ctrl', 'false')

    def test_initialization_custom(self):
        """Test block initializes with custom parameters."""
        with patch.object(_rs_module, 'sdrplay3', mock_sdrplay3):
            blk = rspduo_source(
                frequency=433e6,
                sample_rate=1e6,
                if_gain=50.0,
                rf_gain=10.0,
                bandwidth=1.536e6,
                bias_t=True,
                rf_notch=False,
                dab_notch=False,
                am_notch=True
            )

            src = mock_sdrplay3.rspduo.return_value
            src.set_center_freq.assert_called_with(433e6)
            src.set_sample_rate.assert_called_with(1e6)

            # Verify gain negation
            src.set_gain.assert_any_call(-50.0, 'IF', 0)
            src.set_gain.assert_any_call(-50.0, 'IF', 1)
            src.set_gain.assert_any_call(-10.0, 'RF', 0)
            src.set_gain.assert_any_call(-10.0, 'RF', 1)

            src.set_bandwidth.assert_called_with(1.536e6)

            src.set_setting.assert_any_call('biasT_ctrl', 'true')
            src.set_setting.assert_any_call('rfnotch_ctrl', 'false')
            src.set_setting.assert_any_call('dabnotch_ctrl', 'false')
            src.set_setting.assert_any_call('amnotch_ctrl', 'true')

    def test_two_output_signature(self):
        """Verify hier_block2 is constructed with 2-output io_signature."""
        from gnuradio import gr as mock_gr
        with patch.object(_rs_module, 'sdrplay3', mock_sdrplay3):
            with patch.object(mock_gr, 'io_signature') as mock_io_sig:
                mock_io_sig.return_value = MagicMock()
                blk = rspduo_source()
                # Check that io_signature was called with (2, 2, ...) for output
                calls = mock_io_sig.call_args_list
                # Second call is output signature: io_signature(2, 2, sizeof_gr_complex)
                output_call = calls[1]
                self.assertEqual(output_call[0][0], 2)  # min_streams
                self.assertEqual(output_call[0][1], 2)  # max_streams

    def test_set_frequency(self):
        """Test runtime frequency change."""
        with patch.object(_rs_module, 'sdrplay3', mock_sdrplay3):
            blk = rspduo_source()
            src = mock_sdrplay3.rspduo.return_value
            src.reset_mock()

            blk.set_frequency(915e6)
            src.set_center_freq.assert_called_once_with(915e6)
            self.assertEqual(blk.frequency, 915e6)

    def test_set_sample_rate(self):
        """Test runtime sample rate change."""
        with patch.object(_rs_module, 'sdrplay3', mock_sdrplay3):
            blk = rspduo_source()
            src = mock_sdrplay3.rspduo.return_value
            src.reset_mock()

            blk.set_sample_rate(500e3)
            src.set_sample_rate.assert_called_once_with(500e3)
            self.assertEqual(blk.sample_rate, 500e3)

    def test_set_if_gain_negation(self):
        """Test runtime IF gain change applies negation to both tuners."""
        with patch.object(_rs_module, 'sdrplay3', mock_sdrplay3):
            blk = rspduo_source()
            src = mock_sdrplay3.rspduo.return_value
            src.reset_mock()

            blk.set_if_gain(55.0)
            src.set_gain.assert_any_call(-55.0, 'IF', 0)
            src.set_gain.assert_any_call(-55.0, 'IF', 1)
            self.assertEqual(src.set_gain.call_count, 2)
            self.assertEqual(blk.if_gain, 55.0)

    def test_set_rf_gain_negation(self):
        """Test runtime RF gain change applies negation to both tuners."""
        with patch.object(_rs_module, 'sdrplay3', mock_sdrplay3):
            blk = rspduo_source()
            src = mock_sdrplay3.rspduo.return_value
            src.reset_mock()

            blk.set_rf_gain(20.0)
            src.set_gain.assert_any_call(-20.0, 'RF', 0)
            src.set_gain.assert_any_call(-20.0, 'RF', 1)
            self.assertEqual(src.set_gain.call_count, 2)
            self.assertEqual(blk.rf_gain, 20.0)

    def test_set_bandwidth(self):
        """Test runtime bandwidth change."""
        with patch.object(_rs_module, 'sdrplay3', mock_sdrplay3):
            blk = rspduo_source()
            src = mock_sdrplay3.rspduo.return_value
            src.reset_mock()

            blk.set_bandwidth(1.536e6)
            src.set_bandwidth.assert_called_once_with(1.536e6)

    def test_set_bandwidth_zero_skips(self):
        """Test that setting bandwidth to 0 does not call set_bandwidth."""
        with patch.object(_rs_module, 'sdrplay3', mock_sdrplay3):
            blk = rspduo_source()
            src = mock_sdrplay3.rspduo.return_value
            src.reset_mock()

            blk.set_bandwidth(0)
            src.set_bandwidth.assert_not_called()

    def test_set_bias_t(self):
        """Test runtime Bias-T toggle."""
        with patch.object(_rs_module, 'sdrplay3', mock_sdrplay3):
            blk = rspduo_source()
            src = mock_sdrplay3.rspduo.return_value
            src.reset_mock()

            blk.set_bias_t(True)
            src.set_setting.assert_called_with('biasT_ctrl', 'true')

    def test_set_rf_notch_filter(self):
        """Test runtime RF notch filter toggle."""
        with patch.object(_rs_module, 'sdrplay3', mock_sdrplay3):
            blk = rspduo_source()
            src = mock_sdrplay3.rspduo.return_value
            src.reset_mock()

            blk.set_rf_notch_filter(False)
            src.set_setting.assert_called_with('rfnotch_ctrl', 'false')

    def test_set_dab_notch_filter(self):
        """Test runtime DAB notch filter toggle."""
        with patch.object(_rs_module, 'sdrplay3', mock_sdrplay3):
            blk = rspduo_source()
            src = mock_sdrplay3.rspduo.return_value
            src.reset_mock()

            blk.set_dab_notch_filter(True)
            src.set_setting.assert_called_with('dabnotch_ctrl', 'true')

    def test_set_am_notch_filter(self):
        """Test runtime AM notch filter toggle."""
        with patch.object(_rs_module, 'sdrplay3', mock_sdrplay3):
            blk = rspduo_source()
            src = mock_sdrplay3.rspduo.return_value
            src.reset_mock()

            blk.set_am_notch_filter(True)
            src.set_setting.assert_called_with('amnotch_ctrl', 'true')

    def test_device_settings_error_handling(self):
        """Test that device setting failures don't crash initialization."""
        mock_sdrplay3.rspduo.return_value.set_setting.side_effect = RuntimeError("not supported")
        with patch.object(_rs_module, 'sdrplay3', mock_sdrplay3):
            # Should not raise despite set_setting errors
            blk = rspduo_source(bias_t=True, am_notch=True)
            self.assertIsNotNone(blk)


if __name__ == "__main__":
    unittest.main()
