# GRC YAML Block Definition Practical Spec (block.yml) — Agent-Oriented

This document describes the practical structure of GNU Radio Companion (GRC) YAML block-definition files (`*.block.yml`)
as used in GNU Radio 3.8+ (YAML replaced XML in 3.8).

Primary references:
- GNU Radio wiki: YAML_GRC (living doc; not always exhaustive)
- GNU Radio source: `grc/core/schema_checker/` (enforced truth)
- GNU Radio tracker issues referencing real in-tree block.yml usage patterns (edge-cases)

Goal:
- Make it easy for an automated code agent to understand:
  (1) which keys exist at each hierarchy level,
  (2) how parameters and ports connect,
  (3) how templates drive generated Python/C++ code,
  (4) how to avoid common structural mistakes.

------------------------------------------------------------
0) File identity and scope
------------------------------------------------------------
- File name convention: `<block_id>.block.yml`
- Each file defines one block type (a "block prototype") exposed in the GRC palette.
- The block’s runtime implementation may be Python or C++ (or RFNoC, etc.),
  but this YAML describes the GUI/graph interface + code generation hooks.

------------------------------------------------------------
1) Top-level keys (block object)
------------------------------------------------------------
The YAML root is a mapping (dictionary). Common/expected keys:

REQUIRED (practically always present)
- `id`: string
  Unique block identifier used by GRC and codegen.
  Example: `digital_map_bb`, `qtgui_freq_sink_x`, `blocks_throttle`.

- `label`: string
  Human-readable name in the palette and block header.

- `category`: string or list/structured category path (Observed / likely)
  Determines where the block appears in the GRC block tree.

- `flags`: string | list[string]
  Common values (Observed): `python`, `cpp`, `qt_gui`, `deprecated`, `needs_*`
  Flags are used by GRC to change behavior/availability.

OFTEN PRESENT / IMPORTANT
- `documentation`: string | mapping (Observed / likely)
  May contain:
  - human help text (markdown-ish),
  - or URL/relative URL to documentation.

- `parameters`: list[parameter_object]
  Defines the editable properties shown in the properties dialog, plus how
  those values get injected into templates.

- `inputs`: list[port_object]
  Defines input ports. May be omitted for source-only / variable / control blocks.

- `outputs`: list[port_object]
  Defines output ports. May be omitted for sink-only / variable / GUI sink blocks.

- `templates`: template_object
  Codegen hooks: how to import and instantiate, and how to apply setters (callbacks).

- `file_format`: integer (Observed in docs/issues)
  Used for compatibility/versioning of block.yml structure.

- `asserts`: list[string expressions] (Observed / likely)
  Assertions checked at generate time or validate time (e.g., limits on ninputs).

- `states`: mapping (Observed / likely)
  For advanced blocks that maintain internal GUI state or derived values.

- `value`: string expression (Observed / likely; used in variable-like blocks)
  When a block represents a value (like a Variable/Parameter), this is the expression.

- `var_make`: string expression (Observed / likely; used in variable-like blocks)
  How to emit variable construction / binding for the flowgraph namespace.

- `imports`: list[string] or string (Observed / likely; sometimes inside templates)
  Python import lines required for instantiation.

NOTES
- Some blocks are special “meta” blocks (Options, Variable, Parameter). They often
  omit ports and emphasize `value` / `var_make`.

------------------------------------------------------------
2) `parameters:` list — parameter objects
------------------------------------------------------------
`parameters` is a YAML list. Each entry defines ONE parameter shown in the UI and
bound into codegen.

2.1 Parameter object keys (common)

REQUIRED (practical)
- `id`: string
  Parameter name; referenced by templates via `${id}` or similar.

- `label`: string
  UI label.

- `dtype`: string
  Data type/behavior of the parameter widget and how its value is interpreted.
  Typical (Observed): `string`, `int`, `float`, `bool`, `enum`, `complex`,
  `raw`, `file_open`, `file_save`, `dir_select`, `gui_hint`, `pmt`, etc.
  (Exact set is enforced by schema checker / dtypes module.)

COMMON OPTIONAL
- `default`: scalar | string expression
  Default value shown/used when unset.

- `hide`: enum string or expression resolving to enum
  Controls visibility in UI and/or block label. Common values (Observed in docs/issues):
  - `none`  : show in dialog and on block (if applicable)
  - `part`  : show in dialog, not on block label
  - `all`   : hidden (advanced/derived; not shown)
  IMPORTANT: `hide` is often an expression `${ ... }` that yields one of these strings.

- `category`: string (Observed / likely)
  Used to group parameters in the properties dialog.

- `tab`: string (Observed / likely)
  UI tab grouping in properties dialog (mentioned frequently by users; enforcement in schema checker).

- `options`: list[scalar|string]
  For `dtype: enum` and similar, the legal values.

- `option_labels`: list[string] (Observed / likely)
  Display labels corresponding 1:1 to `options`.

- `base_key`: string (Observed in issues)
  Used when parameters are repeated/derived (e.g., label1/label2 share behavior).

- `gui_hint`: string (Observed / likely)
  Layout hints for Qt GUI parameter dialogs (row/col, stretch, etc. varies by version).

- `callback` / `callbacks`: string | list[string]
  Setter code emitted after instantiation when a parameter changes or on init.
  Often calls `set_*` methods or reconfigures GUI sinks.

- `value`: string expression (Observed / likely)
  A computed parameter value (less common; more common in variable-like blocks).

- `dtype_args`: mapping (Observed / likely)
  Extra config for `dtype` widget behavior (ranges, step size, file filters).

2.2 Parameter evaluation model
- Most editable boxes in GRC are Python expressions (not plain strings) unless the dtype forces string-literal behavior.
- `hide`/`default`/`options` can be expressions, enabling dynamic UI (e.g., number of connections).

2.3 Common pitfalls
- Mismatched lengths: `options` and `option_labels` must align 1:1.
- `dtype: enum` historically restricts variables/expressions in some contexts (tracked in issues).
- If you use expressions in `hide`, ensure they always evaluate to one of {none, part, all}.

------------------------------------------------------------
3) `inputs:` / `outputs:` — port objects
------------------------------------------------------------
Ports define the edges that can be connected in the flowgraph.

Each port entry is a mapping. Common keys:

REQUIRED (typical)
- `domain`: string
  Typical domains:
  - `stream`   : normal streaming IO
  - `message`  : PMT message ports
  - (Observed / likely) `tagged_stream` or other specialized domains depending on GR version.

- `dtype`: string
  For stream ports: element type, e.g. `complex`, `float`, `int`, `short`, `byte`
  For message ports: often `message` or omitted/ignored (implementation-defined).

COMMON OPTIONAL
- `id`: string
  Port identifier name (especially for message ports).

- `label`: string
  User-facing label on the port.

- `vlen`: integer | string expression
  Vector length. For `vlen > 1`, the port item is a vector of dtype.

- `optional`: bool (Observed / likely)
  Whether the port may be left unconnected.

- `multiplicity`: integer | string expression (Observed / likely)
  Used for variable number of ports (e.g. N-input GUI sinks).

- `num_streams` / `nports`: int/expression (Observed / likely)
  Alternate ways to represent port multiplicity; depends on version.

- `stream_id` / `key` (Observed / likely; tagged-stream specific)
  For tagged stream semantics.

Port semantics rules (practical)
- `domain: stream` ports can only connect to other `domain: stream` ports.
- `domain: message` ports can only connect to other `domain: message` ports.
- Stream connections must match dtype AND vlen (unless implicit conversion blocks are used).

------------------------------------------------------------
4) `templates:` — code generation interface
------------------------------------------------------------
`templates` tells GRC how to emit runnable code (Python or C++ flowgraphs).

The `templates` object is typically a mapping with keys like:

COMMON (Observed / likely)
- `imports`: string | list[string]
  Python imports needed (e.g., `from gnuradio import digital`).

- `make`: string
  Constructor expression used to instantiate the block.
  Example (conceptual):
    `digital.map_bb(${map}, ${unused_param})`

- `callbacks`: list[string]
  Setter calls executed after instantiation (and sometimes when changed).
  Example:
    `${id}.set_gain(${gain})`

- `var_make`: string
  For variable-like blocks: how to define variable in generated script.

- `value`: string
  For variable-like blocks: expression representing the value.

- (Advanced) `includes`, `link_libs`, `packages` for C++ workflows (version-dependent)

Generation model
- Parameters from `parameters:` become named symbols available to template rendering.
- GRC expands `${param_id}` expressions during codegen.
- For variable-port blocks, templates often include control logic to create correct port counts.

------------------------------------------------------------
5) `file_format` and compatibility
------------------------------------------------------------
- `file_format` is used to version the YAML structure.
- If absent, older versions may assume a default; modern in-tree blocks usually include it.

------------------------------------------------------------
6) Validation and “spec truth”
------------------------------------------------------------
- The definitive list of allowed keys and their types is enforced by:
  - GNU Radio source: `grc/core/schema_checker/`
- If a key is rejected by schema validation, it is not valid for that GNU Radio version.

Recommended practice for agents/tools
- Treat this doc as a *practical guide*, not an authoritative schema.
- Always validate candidate YAML against the installed GNU Radio version (or CI container)
  using GRC’s built-in validation (load/reload blocks) or a headless check.

------------------------------------------------------------
7) Minimal skeleton example (agent-ready)
------------------------------------------------------------
id: my_block
label: "My Block"
category: "[Custom]/Examples"
flags: [python]

parameters:
  - id: gain
    label: "Gain"
    dtype: float
    default: 1.0
    hide: none

inputs:
  - domain: stream
    dtype: complex
    vlen: 1

outputs:
  - domain: stream
    dtype: complex
    vlen: 1

templates:
  imports: "from my_oot import my_block"
  make: "my_block(${gain})"
  callbacks:
    - "set_gain(${gain})"

file_format: 1

------------------------------------------------------------
8) Edge cases (common in real block.yml)
------------------------------------------------------------
- Dynamic hide:
    hide: ${ 'part' if ${some_condition} else 'all' }

- Multiplicity ports (GUI sinks):
    inputs:
      - domain: stream
        dtype: float
        multiplicity: ${nconnections}

- Message ports:
    inputs:
      - domain: message
        id: "in"
        label: "in"

- Assertions:
    asserts:
      - ${nconnections} <= 10

END
