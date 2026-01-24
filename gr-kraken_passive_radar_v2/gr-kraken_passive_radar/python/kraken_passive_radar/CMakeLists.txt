# Install Python package files
install(FILES
    __init__.py
    DESTINATION ${GR_PYTHON_DIR}/kraken_passive_radar
)

# Add bindings subdirectory
add_subdirectory(bindings)
