# OpenSfM AI Coding Instructions

## 1. Architecture & "Big Picture"
OpenSfM is a Structure-from-Motion library with a hybrid architecture:
- **Core Logic (C++)**: Heavy computational tasks (geometry, bundle adjustment, features) are implemented in C++ under `opensfm/src/`. These are exposed to Python as compiled modules (e.g., `pygeometry`, `pymap`, `pfm`) using **PyBind11**.
- **Pipeline Layout (Python)**: The application logic, pipeline orchestration, and user interface are in Python (`opensfm/` package).
- **Data Abstraction**: The `DataSet` class (`opensfm/dataset.py`) is the central definition for filesystem interactions. All pipeline stages read/write to a hardcoded directory structure (`images/`, `config.yaml`, `reconstruction.json`) managed by this class.
- **State Management**: The `Reconstruction` class (`opensfm/types.py`) wraps the C++ `pymap.Map` object. This pattern (Python class holding a C++ handle) is pervasive.

## 2. Key Developer Workflows

### Building
The project uses `scikit-build-core` to compile C++ extensions.
- **Full Build**: `pip install -e .[test]` (Builds C++ extensions and tests, and installs the package in editable mode).
- **Rebuild C++**: Re-running `pip install -e .` is usually required after changing `.cc` files.
- **Dependencies**: Managed via `pyproject.toml` for managing Python dependencies. Outer envinvironment (C++ toolchain, libraries) is set up through `conda` and the `conda.yml` file. Any change to `conda.yml` requires recreating the conda environment with `conda env create --file conda.yml --yes`, then activating it with `conda activate opensfm`. Note that before running the build commands, the conda environment must be activated once.

### Testing
- **Framework**: `pytest` is used for all tests.
- **Location**: `opensfm/test/` for Python tests. C++ tests are in `./cmake_build`.
- **Synthetic Data**: Python tests heavily rely on synthetic scenes generated in `opensfm/test/conftest.py`. Look for fixtures like `scene_synthetic` or `scene_synthetic_cube`.
- **Run Tests**: `pytest opensfm/test/` for Python tests. C++ tests can be run via `ctest` with `cd cmake_build && ctest --output-on-failure && cd ..`.

### Running the Pipeline
- **Entry Point**: `bin/opensfm` (shell script) -> `bin/opensfm_main.py`.
- **Commands**: Each CLI command (e.g., `detect_features`, `reconstruct`) is implemented as a module in `opensfm/commands/`.
- **Example**: `bin/opensfm reconstruct path/to/dataset` invokes the `run()` method in `opensfm/commands/reconstruct.py`.

## 3. Coding Conventions & Patterns

### Python/C++ Interop
- **Wrappers**: Do not use C++ objects directly if a Python wrapper exists (e.g., use `opensfm.types.Reconstruction` instead of `opensfm.pymap.Map` where possible). 
- **Extensions**: C++ extensions are imported from the root package (e.g., `from opensfm import pygeometry`).

### Type Hinting
- **Strictness**: The codebase uses `pyre-strict`. Ensure all new code has complete type annotations. A comment `# pyre-strict` is often present at the top of files.

### Configuration
- **Definition**: Default parameters are in `opensfm/config.py` (dataclass `OpenSfMConfig`).
- **Overrides**: Parameters are overridden by `config.yaml` in the dataset directory.
- **Access**: Access config values via `data.config['param_name']` (where `data` is a `DataSet` instance).

## 4. Essential Files
- `opensfm/dataset.py`: **READ THIS FIRST** when dealing with file I/O. Defines where every file lives.
- `opensfm/types.py`: Defines key data structures (`Reconstruction`, `Shot`, `Camera`).
- `opensfm/config.py`: Documentation for all tunable parameters.
- `opensfm/src/map/map.cc`: The backing C++ implementation for `Reconstruction`.
- `opensfm/commands/`: Implementation of individual pipeline steps.

## 5. Common Pitfalls
- **Direct File Access**: Avoid manual `open()` calls. Use `dataset.load_*` and `dataset.save_*` methods to ensure consistency with the expected directory structure.
- **Geometry Types**: Be careful with rotation representations (angle-axis vs matrices). `pygeometry` exposes helper functions; check `opensfm/src/geometry/` for implementation details if behavior is unclear.
