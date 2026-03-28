# Heart Rate Estimation using Depth Camera and Pattern Recognition Methods


## Setup
- Camera: Orbbec Femto Bolt
- Ground truth: Movesense medical sensor
- Language: Python 3.12

## Structure
src/
├── recorder/          # Recording script (runs from pyorbbecsdk location)
├── 01_extract_grid.py # Extract depth grid signal from recordings
├── 02_find_chest_tap.py # Synchronize camera and Movesense via chest tap

## Usage
1. Record with `example_recorder.py`
2. Extract grid signal with `01_extract_grid.py`
3. Detect sync tap with `02_find_chest_tap.py`