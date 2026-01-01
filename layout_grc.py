
import re

def parse_yaml_blocks(lines):
    blocks = []
    current_block = []
    in_block = False

    for line in lines:
        if line.strip().startswith('- name:'):
            if current_block:
                blocks.append(current_block)
            current_block = [line]
            in_block = True
        elif in_block:
            if line.strip().startswith('connections:') or line.strip().startswith('metadata:'):
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

def get_block_prop(block_lines, prop):
    # Regex to extract property value
    # Looks for "prop: value"
    for line in block_lines:
        if line.strip().startswith(f"{prop}:"):
            val = line.split(':', 1)[1].strip()
            return val
    return "" # Return empty string instead of None

def set_block_coord(block_lines, x, y):
    new_lines = []
    has_coord = False
    for line in block_lines:
        if line.strip().startswith('coordinate:'):
            indent = line.split('coordinate:')[0]
            new_lines.append(f"{indent}coordinate: [{x}, {y}]\n")
            has_coord = True
        else:
            new_lines.append(line)

    # If coordinate missing, append it? (Shouldn't happen for valid blocks)
    return new_lines

def layout_grc(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header = []
    block_section_raw = []
    footer = []

    section = 'header'
    for line in lines:
        if line.strip().startswith('blocks:'):
            header.append(line)
            section = 'blocks'
            continue
        if line.strip().startswith('connections:'):
            section = 'footer'
            footer.append(line)
            continue

        if section == 'header':
            header.append(line)
        elif section == 'blocks':
            block_section_raw.append(line)
        elif section == 'footer':
            footer.append(line)

    blocks = parse_yaml_blocks(block_section_raw)

    variables = []
    source = []
    filters = []
    dc_blocks = []
    eca = []
    vec_conv = []
    ffts = []
    mults = []
    iffts = []
    dopplers = []
    sinks = []
    others = []

    for b in blocks:
        name = get_block_prop(b, 'name')
        bid = get_block_prop(b, 'id')

        if bid == 'variable':
            variables.append(b)
        elif 'krakensdr' in name:
            source.append(b)
        elif 'eca_canceller' in name:
            eca.append(b)
        elif 'chan' in name and 'filter' in bid:
            filters.append(b)
        elif 'dc' in name or 'blocker' in bid:
            dc_blocks.append(b)
        elif 'stream_to_vector' in bid:
            vec_conv.append(b)
        elif 'fft' in bid and 'ifft' not in bid:
            ffts.append(b)
        elif 'ifft' in bid:
            iffts.append(b)
        elif 'multiply' in bid:
            mults.append(b)
        elif 'doppler' in bid:
            dopplers.append(b)
        elif 'sink' in bid or 'raster' in bid:
            sinks.append(b)
        elif 'vector_to_stream' in bid:
            sinks.append(b)
        else:
            others.append(b)

    ROW_H = 200
    COL_W = 250
    START_X = 8
    START_Y = 150

    processed_blocks = []

    # Variables
    for i, b in enumerate(variables):
        processed_blocks.append(set_block_coord(b, START_X + i*150, 8))

    # Source
    if source:
        processed_blocks.append(set_block_coord(source[0], START_X, START_Y + 200))

    # Filters
    filters.sort(key=lambda x: get_block_prop(x, 'name'))
    # Sorting Ref (r...), Surv (s...) works well enough
    for i, b in enumerate(filters):
         processed_blocks.append(set_block_coord(b, START_X + COL_W, START_Y + i * ROW_H + 200))

    # DC Blocks
    dc_blocks.sort(key=lambda x: get_block_prop(x, 'name'))
    for i, b in enumerate(dc_blocks):
         processed_blocks.append(set_block_coord(b, START_X + 2*COL_W, START_Y + i * ROW_H + 200))

    # ECA
    if eca:
        processed_blocks.append(set_block_coord(eca[0], START_X + 3*COL_W, START_Y + 2.5 * ROW_H + 200))

    def layout_generic(block_list, col_idx):
        block_list.sort(key=lambda x: get_block_prop(x, 'name'))
        for i, b in enumerate(block_list):
            processed_blocks.append(set_block_coord(b, START_X + col_idx*COL_W, START_Y + i * ROW_H + 100))

    layout_generic(vec_conv, 4)
    layout_generic(ffts, 5)
    layout_generic(mults, 6)
    layout_generic(iffts, 7)
    layout_generic(dopplers, 8)
    layout_generic(sinks, 9)
    layout_generic(others, 10)

    with open(filepath, 'w') as f:
        f.writelines(header)
        for b in processed_blocks:
            for line in b:
                f.write(line)
        f.writelines(footer)

if __name__ == "__main__":
    layout_grc('kraken_passive_radar_system.grc')
