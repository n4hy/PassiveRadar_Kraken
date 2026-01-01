
import re
import sys

# --- YAML Parsing Helper ---
def parse_yaml_blocks(lines):
    blocks = []
    current_block = []
    in_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- name:'):
            if current_block:
                blocks.append(current_block)
            current_block = [line]
            in_block = True
        elif in_block:
            if stripped.startswith('connections:') or stripped.startswith('metadata:'):
                in_block = False
                blocks.append(current_block)
                current_block = []
            else:
                current_block.append(line)
        else:
            pass

    if current_block:
        blocks.append(current_block)
    return blocks

def get_prop(block_lines, prop_name):
    for line in block_lines:
        if line.strip().startswith(f"{prop_name}:"):
            val = line.split(':', 1)[1].strip()
            # Remove quotes if present
            return val.strip("'").strip('"')
    return ""

def set_coord(block_lines, x, y):
    new_lines = []
    for line in block_lines:
        if line.strip().startswith('coordinate:'):
            indent = line.split('coordinate:')[0]
            new_lines.append(f"{indent}coordinate: [{x}, {y}]\n")
        else:
            new_lines.append(line)
    return new_lines

def is_debris(name, bid):
    # Debris blocks usually have these generic IDs/Names and connect strictly to other debris
    # The "Good" chain has specific names: ref_*, surv_*, mult_conj*, ifft*, doppler_proc*, vec_to_stream*, rd_raster*

    good_prefixes = [
        'ref_', 'surv_', 'mult_conj', 'ifft', 'doppler_proc',
        'vec_to_stream', 'rd_raster', 'krakensdr', 'eca_canceller',
        'bpf_bw', 'center_freq', 'decim', 'doppler_len', 'eca_taps', 'fft_len', 'rf_gain', 'samp_rate'
    ]

    # Specific known debris patterns from previous analysis
    if name.startswith('blocks_stream_to_vector'): return True
    if name.startswith('fft_vxx'): return True
    if name.startswith('blocks_multiply_conjugate'): return True

    # Allow known good blocks
    for prefix in good_prefixes:
        if name.startswith(prefix):
            return False

    # Default to safe (don't delete unknown blocks unless sure),
    # but based on the file content, the above rules cover the split.
    return False

def get_channel_index(name):
    # Determine vertical row index (0-4)
    # Ref = 0
    # Surv (no suffix) = 1
    # Surv_2 = 2, Surv_3 = 3, Surv_4 = 4
    # Multi-channel blocks like ECA sit in the middle or top (0)

    if 'ref' in name: return 0
    if name.endswith('_4'): return 4
    if name.endswith('_3'): return 3
    if name.endswith('_2'): return 2
    if 'surv' in name: return 1 # Default surv
    return 0

# --- Main Logic ---
def fix_layout(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header = []
    block_section_raw = []
    connections_raw = []
    footer = []

    section = 'header'
    for line in lines:
        if line.strip().startswith('blocks:'):
            header.append(line)
            section = 'blocks'
            continue
        if line.strip().startswith('connections:'):
            connections_raw.append(line)
            section = 'connections'
            continue
        if line.strip().startswith('metadata:'):
            footer.append(line)
            section = 'footer'
            continue

        if section == 'header':
            header.append(line)
        elif section == 'blocks':
            block_section_raw.append(line)
        elif section == 'connections':
            connections_raw.append(line)
        elif section == 'footer':
            footer.append(line)

    blocks = parse_yaml_blocks(block_section_raw)

    # 1. Filter Blocks
    kept_blocks = []
    kept_names = set()

    for b in blocks:
        name = get_prop(b, 'name')
        bid = get_prop(b, 'id')
        if not is_debris(name, bid):
            kept_blocks.append(b)
            kept_names.add(name)
        else:
            print(f"Deleting debris block: {name} ({bid})")

    # 2. Filter Connections
    valid_connections = ['connections:\n']
    for line in connections_raw:
        if '- [' in line:
            # Format: - [source, port, sink, port]
            parts = line.strip().strip('- [').strip(']').split(',')
            src = parts[0].strip().strip("'")
            dst = parts[2].strip().strip("'")

            if src in kept_names and dst in kept_names:
                valid_connections.append(line)

    # 3. Layout
    # Grid Settings
    START_X = 8
    START_Y = 200
    COL_W = 220
    ROW_H = 180

    layout_map = {
        'krakensdr_source': 0,
        'freq_xlating_fir_filter_ccc': 1,
        'dc_blocker_cc': 2,
        'kraken_passive_radar_eca_b_clutter_canceller': 3,
        'stream_to_vector': 4,
        'fft_vcc': 5,
        'multiply_conjugate_vcc': 6,
        'ifft_vcc': 7,
        'kraken_passive_radar_doppler_processing': 8,
        'vector_to_stream': 9,
        'qtgui_time_raster_sink_f': 10
    }

    final_blocks = []
    var_idx = 0

    for b in kept_blocks:
        name = get_prop(b, 'name')
        bid = get_prop(b, 'id')

        if bid == 'variable':
            # Variables at top
            final_blocks.append(set_coord(b, START_X + var_idx*180, 8))
            var_idx += 1
            continue

        # Determine Column
        col = layout_map.get(bid, 11) # Default to end if unknown

        # Determine Row
        row = get_channel_index(name)

        # Special case: ECA Block spans rows, put it in center of its column or top
        if 'eca' in name:
            row = 2 # Center it vertically

        x = START_X + col * COL_W
        y = START_Y + row * ROW_H

        final_blocks.append(set_coord(b, x, y))

    # Write Output
    with open(filepath, 'w') as f:
        f.writelines(header)
        for b in final_blocks:
            for line in b:
                f.write(line)
        f.writelines(valid_connections)
        f.writelines(footer)

    print(f"Layout complete. Kept {len(final_blocks)} blocks.")

if __name__ == "__main__":
    fix_layout('kraken_passive_radar_system.grc')
