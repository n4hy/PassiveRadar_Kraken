#!/bin/bash
# Wrapper script to run unit tests reliably
# Ensures the tests directory is targeted and OOT python path is handled by the tests themselves.
export PYTHONPATH=$PYTHONPATH:$(pwd)/gr-kraken_passive_radar/python
python3 -m unittest discover -s tests -v
