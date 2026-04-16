# Optical Mark Recognition (OMR) System

A robust, high-accuracy end-to-end Optical Mark Recognition system designed for automated grading. This project provides both a rapid-development Python pipeline and a high-performance C++ core engine.

For a deep dive into the algorithmic framework and methods, see [METHODOLOGY.md](METHODOLOGY.md).

## 🚀 Quick Start

### 1. Python Pipeline (R&D & Testing)

**Prerequisites:**
- Python 3.9+
- `uv` package manager (or `pip`)

**Setup & Execution:**
```bash
# 1. Sync dependencies and activate environment
uv sync
source .venv/bin/activate

# 2. Run the End-to-End pipeline
uv run (or python) experiments/end2end.py
```
*This will parse the input image, score against `test/keys/TEST_1.csv`, and output a scaled grading report in `outputs/scanner_debug_ver2/final_results.csv`.*

### 2. C++ Core Engine (Production module)

**Prerequisites:**
- OpenCV 4 (`libopencv-dev`)
- CMake 3.10+
- A C++17 compliant compiler (GCC/Clang)

**Setup & Execution (Ubuntu/WSL):**
```bash
# 1. Install required packages
sudo apt update
sudo apt install libopencv-dev cmake -y

# 2. Build the engine
cd core
mkdir -p build && cd build
cmake ..
make

# 3. Run the processing unit
./omr_engine
```
*The engine will output `final_results_cpp.csv` with identical algorithmic precision to the Python version.*

## 📁 Project Structure highlights
- `experiments/`: R&D Python scripts (anchor detection, OCR logic, shape extraction).
- `core/`: C++ production engine with parallel logic and CMake layout.
- `test/`: Evaluation sets, keys (CSV), and layout templates (JSON).
- `outputs/`: Warped images, debugging visual outputs, and CSV reports.