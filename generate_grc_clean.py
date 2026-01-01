
import sys

def create_block(name, id, parameters, coordinate, rotation=0, state='enabled'):
    return {
        'name': name,
        'id': id,
        'parameters': parameters,
        'states': {
            'bus_sink': False,
            'bus_source': False,
            'bus_structure': None,
            'coordinate': coordinate,
            'rotation': rotation,
            'state': state
        }
    }

def format_yaml(data):
    output = []

    # Options Block
    output.append("options:")
    output.append("  parameters:")
    for k, v in data['options']['parameters'].items():
        val = v
        # Simple heuristic to avoid quoting unnecessary things, but GRC is forgiving
        output.append(f"    {k}: {val}")
    output.append("  states:")
    output.append("    bus_sink: false")
    output.append("    bus_source: false")
    output.append("    bus_structure: null")
    output.append(f"    coordinate: [{data['options']['states']['coordinate'][0]}, {data['options']['states']['coordinate'][1]}]")
    output.append(f"    rotation: {data['options']['states']['rotation']}")
    output.append(f"    state: {data['options']['states']['state']}")
    output.append("")

    # Blocks
    output.append("blocks:")
    for b in data['blocks']:
        output.append(f"- name: {b['name']}")
        output.append(f"  id: {b['id']}")
        output.append("  parameters:")
        for k, v in b['parameters'].items():
            val = v
            if val == '': val = "''"
            output.append(f"    {k}: {val}")
        output.append("  states:")
        output.append("    bus_sink: false")
        output.append("    bus_source: false")
        output.append("    bus_structure: null")
        output.append(f"    coordinate: [{b['states']['coordinate'][0]}, {b['states']['coordinate'][1]}]")
        output.append(f"    rotation: {b['states']['rotation']}")
        output.append(f"    state: {b['states']['state']}")

    # Connections
    output.append("connections:")
    for c in data['connections']:
        output.append(f"- [{c[0]}, '{c[1]}', {c[2]}, '{c[3]}']")

    output.append("")
    # Metadata
    output.append("metadata:")
    output.append("  file_format: 1")
    output.append("  grc_version: 3.10.9.2")

    return "\n".join(output)

def generate():
    START_X = 8
    START_Y = 200
    COL_W = 250
    ROW_H = 200

    blocks = []
    connections = []

    # --- Variables ---
    vars = [
        ('samp_rate', '2048000'),
        ('center_freq', '99.5e6'),
        ('rf_gain', '10'),
        ('fft_len', '4096'),
        ('doppler_len', '128'),
        ('decim', '2'),
        ('bpf_bw', '180e3'),
        ('eca_taps', '16')
    ]

    for i, (name, val) in enumerate(vars):
        blocks.append(create_block(
            name, 'variable', {'value': val, 'comment': ''},
            [START_X + i*150, 8]
        ))

    # --- Source ---
    blocks.append(create_block(
        'krakensdr_source_0', 'kraken_passive_radar_krakensdr_source',
        {
            'frequency': 'center_freq',
            'gain': 'rf_gain',
            'sample_rate': 'samp_rate',
            'comment': ''
        },
        [START_X, START_Y + 2*ROW_H]
    ))

    # --- Per-Channel Processing ---
    channels = ['ref', 'surv_0', 'surv_1', 'surv_2', 'surv_3']

    # Columns
    C_FILT = 1
    C_DC = 2
    C_ECA = 3
    C_S2V = 4
    C_FFT = 5
    C_MULT = 6
    C_IFFT = 7
    C_DOP = 8
    C_V2S = 9
    C_SINK = 10

    # ECA Block
    eca_name = 'eca_canceller'
    blocks.append(create_block(
        eca_name, 'kraken_passive_radar_eca_b_clutter_canceller',
        {
            'num_taps': 'eca_taps',
            'num_surv_channels': '4',
            'lib_path': "''",
            'comment': ''
        },
        [START_X + C_ECA*COL_W, START_Y + 2*ROW_H]
    ))

    for i, ch_name in enumerate(channels):
        row_y = START_Y + i*ROW_H

        # Suffix/Naming logic
        if i == 0:
            src_port = 0
            filt_name = "ref_chan"
            dc_name = "ref_dc"
            s2v_name = "ref_vec"
            fft_name = "ref_fft"
            mult_name = "mult_conj"
            ifft_name = "ifft"
            dop_name = "doppler_proc"
            v2s_name = "vec_to_stream"
            sink_name = "rd_raster"
        else:
            src_port = i
            # For Surv
            # Ensure unique names for all blocks in the chain for channels 1..4
            # Using simple _i suffix for all (where i=1,2,3,4) is safest and clearest.

            # Special handling for "Surv 1" to match legacy if desired,
            # but cleaner to just be consistent.
            # Legacy often uses "name" then "name_0" or "name_1".
            # Let's use strict numbering based on 'i'.

            if i == 1:
                # To match "surv_chan" (no number) often seen?
                # The previous code used "surv_chan" for i=1.
                filt_name = "surv_chan"
                dc_name = "surv_dc"
                s2v_name = "surv_vec"
                fft_name = "surv_fft"
                # mult_conj_1, ifft_1, etc.
                mult_name = f"mult_conj_{i}"
                ifft_name = f"ifft_{i}"
                dop_name = f"doppler_proc_{i}"
                v2s_name = f"vec_to_stream_{i}"
                sink_name = f"rd_raster_{i}"
            else:
                filt_name = f"surv_chan_{i}"
                dc_name = f"surv_dc_{i}"
                s2v_name = f"surv_vec_{i}"
                fft_name = f"surv_fft_{i}"
                mult_name = f"mult_conj_{i}"
                ifft_name = f"ifft_{i}"
                dop_name = f"doppler_proc_{i}"
                v2s_name = f"vec_to_stream_{i}"
                sink_name = f"rd_raster_{i}"

        # 1. Filter
        blocks.append(create_block(
            filt_name, 'freq_xlating_fir_filter_ccc',
            {
                'decim': 'decim',
                'taps': 'firdes.low_pass(1.0, samp_rate, bpf_bw/2, bpf_bw/4, firdes.WIN_HAMMING)',
                'center_freq': '0',
                'sample_rate': 'samp_rate',
                'comment': ''
            },
            [START_X + C_FILT*COL_W, row_y]
        ))
        connections.append(('krakensdr_source_0', str(src_port), filt_name, '0'))

        # 2. DC Block
        blocks.append(create_block(
            dc_name, 'dc_blocker_cc',
            {'length': '128', 'long_form': 'True', 'comment': ''},
            [START_X + C_DC*COL_W, row_y]
        ))
        connections.append((filt_name, '0', dc_name, '0'))

        # Connect DC to ECA
        connections.append((dc_name, '0', eca_name, str(i)))

        # --- Post-ECA Chain ---

        # 3. Stream to Vector
        blocks.append(create_block(
            s2v_name, 'stream_to_vector',
            {'type': 'complex', 'num_items': 'fft_len', 'vlen': '1', 'comment': ''},
            [START_X + C_S2V*COL_W, row_y]
        ))
        # Connection from ECA Output
        connections.append((eca_name, str(i), s2v_name, '0'))

        # 4. FFT
        blocks.append(create_block(
            fft_name, 'fft_vcc',
            {
                'fft_size': 'fft_len', 'forward': 'True', 'shift': 'True',
                'window': 'window.blackmanharris(fft_len)', 'nthreads': '1', 'comment': ''
            },
            [START_X + C_FFT*COL_W, row_y]
        ))
        connections.append((s2v_name, '0', fft_name, '0'))

        # 5. Multiply Conjugate
        blocks.append(create_block(
            mult_name, 'multiply_conjugate_vcc',
            {'vlen': 'fft_len', 'comment': ''},
            [START_X + C_MULT*COL_W, row_y]
        ))

        connections.append((fft_name, '0', mult_name, '0'))
        if i == 0:
            connections.append((fft_name, '0', mult_name, '1'))
        else:
            connections.append(('ref_fft', '0', mult_name, '1'))

        # 6. IFFT
        blocks.append(create_block(
            ifft_name, 'ifft_vcc',
            {
                'fft_size': 'fft_len', 'forward': 'True', 'shift': 'True',
                'window': 'window.blackmanharris(fft_len)', 'nthreads': '1', 'comment': ''
            },
            [START_X + C_IFFT*COL_W, row_y]
        ))
        connections.append((mult_name, '0', ifft_name, '0'))

        # 7. Doppler Processing
        blocks.append(create_block(
            dop_name, 'kraken_passive_radar_doppler_processing',
            {'fft_len': 'fft_len', 'doppler_len': 'doppler_len', 'comment': ''},
            [START_X + C_DOP*COL_W, row_y]
        ))
        connections.append((ifft_name, '0', dop_name, '0'))

        # 8. Vector to Stream
        blocks.append(create_block(
            v2s_name, 'vector_to_stream',
            {'type': 'float', 'num_items': 'fft_len * doppler_len', 'vlen': '1', 'comment': ''},
            [START_X + C_V2S*COL_W, row_y]
        ))
        connections.append((dop_name, '0', v2s_name, '0'))

        # 9. Raster Sink
        ch_label = f"Ch{i}" if i > 0 else "Ref"
        blocks.append(create_block(
            sink_name, 'qtgui_time_raster_sink_f',
            {
                'name': f"'Range-Doppler (CAF) {ch_label}'",
                'nrows': 'doppler_len', 'ncols': 'fft_len',
                'x_axis_label': "'Range Bins'", 'y_axis_label': "'Doppler Bins'",
                'zmin': '0', 'zmax': '100', 'grid': 'False', 'update_time': '0.10',
                'comment': ''
            },
            [START_X + C_SINK*COL_W, row_y]
        ))
        connections.append((v2s_name, '0', sink_name, '0'))

    # Construct final data dict
    data = {
        'options': {
            'parameters': {
                'id': 'kraken_passive_radar_system',
                'title': 'KrakenSDR Passive Radar',
                'author': '',
                'description': 'KrakenSDR Passive Radar System (ECA-B + CAF)',
                'window_size': '1280,800',
                'generate_options': 'qt_gui',
                'category': 'Custom',
                'run_options': 'prompt',
                'run': 'True',
                'max_nouts': '0',
                'realtime_scheduling': '',
                'cmake_opt': '',
                'gen_cmake': 'On',
                'gen_linking': 'dynamic',
                'output_language': 'python',
                'catch_exceptions': 'True',
                'placement': '(0,0)',
                'qt_qss_theme': '',
                'thread_safe_setters': '',
                'sizing_mode': 'fixed',
                'run_command': '{python} -u {filename}',
                'hier_block_src_path': '.:',
                'comment': '',
                'copyright': ''
            },
            'states': {
                'coordinate': [8, 8],
                'rotation': 0,
                'state': 'enabled'
            }
        },
        'blocks': blocks,
        'connections': connections
    }

    with open('kraken_passive_radar_system.grc', 'w') as f:
        f.write(format_yaml(data))

    print("GRC file generated successfully.")

if __name__ == "__main__":
    generate()
