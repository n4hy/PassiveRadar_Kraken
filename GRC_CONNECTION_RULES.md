# GRC_CONNECTION_RULES_AND_NORMAL_FORM.md
# GNU Radio Companion connection capability rule set + port normal form (agent-oriented)

Scope:
- This rule set is about what *may* be connected in GRC at the graph level.
- It is intentionally “mechanical”: a coding agent should be able to implement it with no interpretation.

Terminology:
- A **port** is an endpoint on a block: input or output.
- A **connection** is a directed edge: (src_block.src_port) -> (dst_block.dst_port).
- Ports have a **domain**. At minimum: `stream` and `message`.
- Stream ports have an **item type** (`dtype`) and an optional **vector length** (`vlen`).

----------------------------------------------------------------------
1) Connection capability rule set (hard constraints)
----------------------------------------------------------------------

1.1 Domain compatibility (MUST match)
- A `stream` output MAY connect only to a `stream` input.
- A `message` output MAY connect only to a `message` input.
- Cross-domain connections are invalid:
  - `stream -> message` invalid
  - `message -> stream` invalid

Rationale:
- Stream connections represent typed sample streams.
- Message connections represent PMT message events.

1.2 Directionality (MUST be output -> input)
- A connection MUST start at an output port and end at an input port.
- `input -> input` and `output -> output` are invalid.

1.3 Stream dtype compatibility (MUST match unless there is an explicit converter block)
For `domain == stream`:
- Let `dtype_out` be the source stream dtype and `dtype_in` be the destination stream dtype.
- A direct connection is valid IFF: `dtype_out == dtype_in`.

No implicit casting:
- `float -> complex` is invalid without an explicit `float_to_complex` or equivalent converter.
- `short -> float` is invalid without `short_to_float`.
- `complex -> float` is invalid without `complex_to_mag`, `complex_to_real`, etc.

1.4 Stream vector length compatibility (MUST match unless there is an explicit reshape/interleave/deinterleave block)
For `domain == stream`:
- Let `vlen_out` be source vlen, `vlen_in` destination vlen.
- Normalize missing vlen to 1 (scalar items).
- A direct connection is valid IFF: `vlen_out == vlen_in`.

No implicit packing/unpacking:
- `vlen 1 -> vlen N` invalid unless you insert a block that packs/streams-to-vector.
- `vlen N -> vlen 1` invalid unless you insert a block that unpacks/vector-to-stream.
- `vlen M -> vlen N` invalid unless explicit reshape exists (rare and typically explicit).

1.5 Stream itemsize equivalence (derived; MUST match if dtype and vlen match)
If you maintain an itemsize table (recommended):
- itemsize = sizeof(dtype) * vlen
- For a direct connection: itemsize_out == itemsize_in is implied by dtype/vlen equality.
- Use itemsize as a fast check or to catch mismatched dtype aliases.

1.6 Message port identity (MUST satisfy port existence; id naming does not need to match)
For `domain == message`:
- A message output may connect to any message input.
- Port `id` strings do not need to match; they are local to the block.
- The only hard constraint is domain equality and port existence.

1.7 Port multiplicity / fanout / fanin constraints (cardinality rules)
These are per-port properties, not global.

A) Stream outputs
- Stream outputs may generally **fan out** to multiple stream inputs (one-to-many) unless the runtime block prohibits it.
- In GNU Radio runtime, fanout is supported: one output buffer can feed multiple downstream blocks.
- Therefore default rule: stream output fanout is allowed.

B) Stream inputs
- Stream inputs generally accept **at most one** upstream stream connection per input index.
- Default rule: a given stream input port index must have 0 or 1 upstream connection.
- If a block supports “multiple connections into one logical input,” it is represented as multiple input ports, not multiple edges into one port.

C) Message outputs
- Message outputs may fan out to multiple message inputs (publish/subscribe style).

D) Message inputs
- Message inputs may accept at most one direct connection per input *port* in many UIs, but message routing blocks often provide multiple inputs explicitly.
- Conservative default for an agent: 0 or 1 incoming edge per message input port.

E) Multiplicity / variable-port blocks
- If a port has `multiplicity`, `nports`, or equivalent:
  - The block effectively has a *port family* with indices [0..K-1], where K = evaluated multiplicity.
  - Each indexed stream input still obeys the “0 or 1 incoming edge” rule.
  - Each indexed output may fan out.

1.8 Tagged stream and tagged-stream domains (common in GR, treated conservatively)
Some systems distinguish:
- `stream` (untagged semantics)
- `tagged_stream` (packet-like segments with a length tag)

Agent rule (safe default):
- Treat `tagged_stream` as a distinct domain that MUST match exactly:
  - `tagged_stream -> tagged_stream` valid (subject to dtype/vlen)
  - `tagged_stream -> stream` invalid unless an explicit converter exists
  - `stream -> tagged_stream` invalid unless an explicit converter exists
If your installed GNU Radio collapses these into `stream`, you can relax this rule.

1.9 Optional ports
- If a port is marked `optional: true`, it may be left unconnected with no error.
- If not optional, leaving unconnected may cause GRC validation errors or runtime issues.
- Optional does not change compatibility rules; it changes whether “no connection” is permitted.

----------------------------------------------------------------------
2) “Normal form” for port signatures (canonical representation)
----------------------------------------------------------------------

Purpose:
- Agents frequently fail because they do not compare ports in a consistent representation.
- This normal form removes ambiguity and makes compatibility checks trivial.

2.1 PortSignature normal form (per port)
Represent every port as the tuple:

PortSignature :=
  (domain, dtype, vlen, msg_type, optional, family_id, index)

Where:
- domain: one of {"stream", "message", "tagged_stream", ...}
- dtype: for stream ports: one of {"complex", "float", "int", "short", "byte", ...}
        for message ports: set to "pmt" (or "message") as a placeholder
- vlen: integer >= 1; default 1 for stream; set 1 for message
- msg_type: optional refinement for message ports (often unknown); default "pmt_any"
- optional: boolean; default false if missing
- family_id: string identifying a multiplicity group; default to port id or label
- index: integer >= 0 for port families; 0 for non-multiple ports

2.2 BlockInterface normal form (per block)
BlockInterface :=
  {
    block_id: string,
    inputs:  [PortSignature ...]  // fully expanded, indexed ports
    outputs: [PortSignature ...]  // fully expanded, indexed ports
    constraints: {
      // optional extra constraints your agent can enforce
      max_in_edges_per_input: 1 (default),
      allow_stream_fanout: true (default),
      allow_msg_fanout: true (default)
    }
  }

2.3 Expansion rule for multiplicity
If a port object has multiplicity `K`:
- Expand into K ports with indices 0..K-1, same domain/dtype/vlen, same family_id.

Example:
inputs:
  - domain: stream
    dtype: complex
    multiplicity: 4

Normal form expands to:
inputs = [
  ("stream","complex",1,"pmt_any",false,"in",0),
  ("stream","complex",1,"pmt_any",false,"in",1),
  ("stream","complex",1,"pmt_any",false,"in",2),
  ("stream","complex",1,"pmt_any",false,"in",3)
]

2.4 Compatibility predicate in normal form
Given src output port S and dst input port D in normal form:

Compatible(S, D) is true IFF:
- S.domain == D.domain
- and if domain is stream (or tagged_stream treated as stream-like):
    S.dtype == D.dtype
    AND S.vlen == D.vlen
- and if domain is message:
    true (msg_type ignored unless you have extra info)

Then apply cardinality checks:
- Each input port index can have at most 1 upstream edge (default).
- Output fanout allowed by default.

----------------------------------------------------------------------
3) Extension: explicit connection algorithm for dumb agents
----------------------------------------------------------------------

The following procedure is intended to be directly implementable.

Inputs:
- A set of blocks with their BlockInterface normal forms.
- A proposed edge (src_block, src_output_index) -> (dst_block, dst_input_index).

Algorithm:

A) Validate endpoints exist
1. Confirm src_output_index within src.outputs length.
2. Confirm dst_input_index within dst.inputs length.

B) Check domain and type compatibility
3. Let S = src.outputs[src_output_index], D = dst.inputs[dst_input_index].
4. If S.domain != D.domain: reject.
5. If S.domain in {"stream","tagged_stream"}:
     if S.dtype != D.dtype: reject.
     if S.vlen != D.vlen: reject.
6. If S.domain == "message": accept type-wise.

C) Check input cardinality
7. If dst input port already has an incoming edge: reject (default).
   (Unless you explicitly model a rare “multi-edge input” feature; most blocks do not.)

D) Record edge and update fanout counters
8. Add edge.
9. Optionally enforce output fanout limits if a block declares them (default unlimited).

E) Report required converters if incompatible (agent helpful mode)
10. If rejected due to dtype mismatch:
      Suggest an explicit converter block based on (S.dtype -> D.dtype) map.
    If rejected due to vlen mismatch:
      Suggest stream-to-vector, vector-to-stream, interleave/deinterleave, or repack blocks.

----------------------------------------------------------------------
4) Converter suggestion map (minimal, pragmatic)
----------------------------------------------------------------------

This is NOT a guarantee of block names across versions; it’s a suggestion heuristic.
Map is keyed by (dtype_out, dtype_in).

Examples:
- (float -> complex): "float_to_complex" (or build complex from real/imag)
- (complex -> float): "complex_to_real" OR "complex_to_mag" OR "complex_to_mag_squared"
- (short -> float): "short_to_float"
- (byte -> float): "uchar_to_float" / "char_to_float" depending on signedness
- (float -> short): "float_to_short" with scaling considerations
- (vector length mismatch): "stream_to_vector" and "vector_to_stream"
- (complex scalar -> complex vlen N): "stream_to_vector" then pass through
- (complex vlen N -> complex scalar): "vector_to_stream"

Agent guidance:
- If signedness ambiguity exists (byte vs char vs uchar), require human confirmation or inspect upstream block definitions.

----------------------------------------------------------------------
5) Edge cases your agent should explicitly NOT “invent”
----------------------------------------------------------------------

- Do not assume implicit type casting exists.
- Do not assume implicit vlen reshape exists.
- Do not assume “rate compatibility” is required for a connection. In GNU Radio:
  - sample rate mismatches are a *system correctness* concern but not a port-type connection constraint.
- Do not assume message payload types. Treat all PMT as compatible unless you have explicit metadata.

END
