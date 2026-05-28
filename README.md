# Heart Rate Estimation using Depth Camera and Pattern Recognition Methods


# Setup
- **Camera**: Orbbec Femto Bolt (Time-of-Flight depth sensor)
- **Ground truth**: Movesense medical chest strap (ECG-based)
- **Language**: Python 3.12
- **Dataset**: 76 recordings, 13 subjects, 3 distances (800/1200/1800mm), 2 clothing types (t-shirt/hoodie)
- **Reference paper**: Rong & Bliss, *"Insights on Using Time-of-Flight Camera for Recovering Cardiac Pulse from Chest Motion in Depth Videos"*, IEEE TBME 2024.


## Structure
```
src/
├── recorder/                       # Recording script (runs from pyorbbecsdk location)
├── 01_extract_grid.py              # Extract depth grid signal from recordings
├── 02_find_chest_tap.py            # Synchronize camera and Movesense via chest tap
├── 03_filter_grid_cells.py         # Bandpass filter (0.5-4 Hz) + trim post-tap
├── 04_ecg_preprocessing.py         # Preprocess Movesense signal NOT YET DONE
├── 05_rpca.py                      # RPCA decomposition with IALM
├── 06_eigenvectors.py              # MAIN PIPELINE: windowed RPCA + eigenvector selection
                                    #   + parabolic interp + comb refinement + smoothing

```
## Pipeline (in order)
 
1. **Record** with `recorder/example_recorder.py` → produces `D:\recordings\{REC_ID}\`
2. **Extract grid** with `01_extract_grid.py` → produces `GRID_{REC_ID}.csv`
3. **Detect sync tap** with `02_find_chest_tap.py` → produces `Tap_info_{REC_ID}.json`
4. **Bandpass filter** with `03_filter_grid_cells.py` → produces `FILTERED_{REC_ID}.csv`
5. **Estimate HR** with `06_eigenvectors.py {REC_ID}` → produces `selection_windowed_{REC_ID}.json` -> PULSE_FILES
6. **Batch all recordings** with `batch/batch_eigenvectors.py`
7. **Aggregate results** with `summary_table.py` + `summary_graph.py`

## Current Results
- Mean absolute error: **5.13 BPM** (76 recordings)
- Median error: **4.68 BPM**
- T-shirt only: 4.92 BPM mean / 3.55 BPM median (38 recordings)
- Best condition (t-shirt + 1200mm): 3.41 BPM mean / **2.70 BPM median**
See `PIPELINE.md` for detailed per-script documentation (inputs, outputs, what each script does).
 