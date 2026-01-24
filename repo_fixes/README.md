# Repository Fixes for gr-kraken_passive_radar OOT Module

These files correct three bugs discovered during build/test:

## File 1: `CMakeLists.txt` (top-level)

**Problem:** Python bindings installed to wrong location (`/usr/local/lib/python3/...`) 
instead of gnuradio's namespace (`/usr/lib/python3/dist-packages/gnuradio/`).

**Fix:** Changed Python path detection from generic `sysconfig` to finding where 
gnuradio is actually installed.

**Location in repo:** `gr-kraken_passive_radar/CMakeLists.txt`


## File 2: `python/kraken_passive_radar/__init__.py`

**Problem:** `ImportError: generic_type: type "eca_canceller" referenced unknown base type "gr::sync_block"`

**Fix:** Import `gnuradio.gr` before importing our pybind11 module so Python knows 
about the base types.

**Location in repo:** `gr-kraken_passive_radar/python/kraken_passive_radar/__init__.py`


## File 3: `python/kraken_passive_radar/bindings/eca_canceller_python.cc`

**Problem:** pybind11 binding explicitly referenced `gr::sync_block`, `gr::block`, 
`gr::basic_block` as base classes, but these weren't registered with pybind11.

**Fix:** Removed explicit base class specification from `py::class_<>` template.

**Location in repo:** `gr-kraken_passive_radar/python/kraken_passive_radar/bindings/eca_canceller_python.cc`


## File 4: `grc/kraken_passive_radar_eca_canceller.block.yml`

**Problem:** `FlowGraph Error: ${num_taps} > 0 is not a mako substitution`

**Fix:** The `asserts:` section uses raw Python expressions, not Mako `${}` syntax.
Changed `${num_taps} > 0` to `num_taps > 0`.

**Location in repo:** `gr-kraken_passive_radar/grc/kraken_passive_radar_eca_canceller.block.yml`


## Installation

Copy these files to your repository, replacing the originals:

```bash
cp repo_fixes/CMakeLists.txt gr-kraken_passive_radar/
cp repo_fixes/grc/kraken_passive_radar_eca_canceller.block.yml gr-kraken_passive_radar/grc/
cp repo_fixes/python/kraken_passive_radar/__init__.py gr-kraken_passive_radar/python/kraken_passive_radar/
cp repo_fixes/python/kraken_passive_radar/bindings/eca_canceller_python.cc gr-kraken_passive_radar/python/kraken_passive_radar/bindings/
```

Then rebuild:

```bash
cd gr-kraken_passive_radar
rm -rf build && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

Verify:

```bash
python3 -c "from gnuradio import kraken_passive_radar; print(dir(kraken_passive_radar))"
```
