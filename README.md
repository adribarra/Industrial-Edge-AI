# Industrial Edge AI

A small project that looks at pictures of screws and tells you if each one is OK (PASS) or
defective (FAIL). It uses the MVTec AD dataset, only the `screw` category.

Right now everything runs on a normal computer CPU. The idea is to later move it to an Altera
Agilex 5 FPGA board with the FPGA AI Suite, but that part is not done yet.

I developed and tested this on an Intel Mac with Python 3.12. I have not tried it on Windows yet.

## What it does

1. Reads the screw images from the dataset folder.
2. Resizes and normalizes each image.
3. Runs it through a model that has only ever seen good screws.
4. Compares the model's score with a threshold. High score means FAIL, low score means PASS.
5. Measures how fast this is and saves two report files.

The app talks to the model through a simple interface (`inference/base.py`). That way a
different backend, like the AI Suite one, can be plugged in later without rewriting the app.
The AI Suite file is only a placeholder for now and just raises an error.

### The model

It is a simplified version of PatchCore using a ResNet18 that was pretrained on ImageNet.
Nothing is trained by gradient descent. The model just remembers small pieces ("patches") of
good screw images. For a new image, it checks how far each patch is from the closest piece it
remembers. The biggest distance becomes the anomaly score, so a higher score means the image
looks more unusual. Input size is 256x256.

### How I kept the data separate

The dataset has 320 good training images. I used 288 to build the model's memory and kept 32
aside to choose the threshold. The test images were never used for either of those.

## Install

macOS or Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The versions in `requirements.txt` are pinned (for example `numpy<2` and `torch==2.2.2`)
because those are the newest ones that work on an Intel Mac. Other systems may work with newer
versions.

## Get the data and the model (once, about 250 MB)

```bash
python scripts/download_dataset.py
python scripts/build_memory_bank.py
python scripts/calibrate_threshold.py
```

The first one downloads the screw images (about 195 MB, license CC BY-NC-SA 4.0, so
non-commercial use only). The second downloads the ResNet18 weights (about 47 MB) and builds the
memory, which takes around 40 seconds on my machine. The third picks the threshold. None of
these files are stored in git. See `datasets/README.md` and `models/README.md` for sources and
licenses.

## Run it

```bash
python -m app.main --dataset ./datasets/mvtec --category screw --backend cpu --output ./results
```

Some options you can use: `--limit N` to only process the first N images, `--threshold X` to
set your own threshold, `--warmup N`, `--split train|test` and `--model-dir DIR`.

It writes `results/inspection_results.csv` and `results/benchmark.json`.

### Example (real run, 160 test images)

```
Inspecting 160 test images of 'screw' (backend=cpu, threshold=2.1465) ...
========================================================
Inspection summary
========================================================
Images processed : 160 (0 errors)
PASS / FAIL      : 135 / 25
Anomalous found  : 25/119 (missed 94)
False alarms     : 0/41
Accuracy         : 0.412
--------------------------------------------------------
Benchmark (cpu, device=cpu, patchcore-resnet18)
  Load time      : 213.8 ms
  Latency mean   : 370.6 ms
  Latency median : 316.5 ms
  Latency P95    : 613.6 ms
  Throughput     : 2.70 images/s
  Warm-up/measured: 3/160
  NOTE: CPU measurement only.
========================================================
```

The first lines of `inspection_results.csv`:
```
image_path,expected_label,predicted_label,anomaly_score,decision,latency_s,error
datasets/mvtec/screw/test/good/000.png,normal,normal,1.5710,PASS,0.2939,
datasets/mvtec/screw/test/good/001.png,normal,normal,1.7314,PASS,0.2830,
```

`benchmark.json` has the model name, backend, device, category, number of images, average,
median, P95, min and max latency, throughput, total time, model load time and the settings used
for the measurement.

### What these results really mean

- The accuracy is not good. The threshold is the highest score among only 32 good validation
  images, so it is very strict. It never flags a good screw, but it misses 94 of the 119
  defective ones.
- The model does separate good from bad to some degree (AUROC around 0.84), but `screw` is one
  of the harder categories and the two groups overlap a lot.
- A lower threshold would catch more defects but also flag some good screws. It should be
  chosen on validation data, never on the test data.
- All speed numbers are for the CPU on this one machine. They say nothing about how fast an
  FPGA would be, and I would not compare them with FPGA numbers unless both were measured the
  same way.
- The timing changes from run to run (P95 was 312 ms in a short 30-image run and 614 ms in the
  full run). I did not look into why.
- The model also produces a map of where the anomaly is, but I have not checked how good that
  map is.

## Tests

```bash
pytest
```

Currently 43 tests pass in about 15 seconds.

| File | Tests | What it checks |
|---|---|---|
| `test_dataset.py` | 5 | listing images, labels, masks, missing folders |
| `test_preprocessing.py` | 11 | image shape, data type, normalization, color order, errors |
| `test_backend.py` | 4 | the backend interface and the AI Suite placeholder |
| `test_cpu_backend.py` | 5 | backend behavior with a dummy network, plus one real image (skipped if the model or data is missing) |
| `test_inspection.py` | 6 | threshold logic, keeping the original score, errors, calibration |
| `test_benchmark.py` | 5 | warm-up, statistics, load time kept separate |
| `test_reporting.py` | 4 | CSV, JSON and the terminal summary |
| `test_cli.py` | 3 | default arguments and error exits |

The tests use tiny made-up images and fake backends, so they do not need the dataset. The fake
backends are only for testing and are not real models.

## Verilog streaming module

`rtl/image_stream.v` is a small one-stage register that passes data along using a valid/ready
handshake. It has a clock, a synchronous reset, data in, `valid`, `ready` and a `last` signal
that marks the end of a frame. Data only moves when `valid` and `ready` are both high. If the
output is not ready, the input is told to wait, so nothing gets lost or overwritten.

This is a separate learning exercise. It is not connected to the AI Suite or to the Python
application.

`rtl/tb_image_stream.v` is a testbench that checks itself. It keeps a list of what went in and
makes sure the same data and `last` flags come out in the same order. It covers reset, a single
transfer, back-to-back transfers, frame end, output stalls with backpressure, a reset in the
middle of a stream, and 400 cycles of random traffic. To run it with Icarus Verilog:

```bash
cd rtl
iverilog -g2005 -Wall -o /tmp/tb.vvp image_stream.v tb_image_stream.v
vvp /tmp/tb.vvp
```

Result:
```
SUMMARY: 249 input transfers, 248 output transfers, 29 frames ends, 0 errors
PASS
```

The one missing transfer is expected, because the mid-stream reset throws away one item. To
make sure the testbench can actually fail, I broke the module on purpose in three ways
(ignoring backpressure, dropping `last`, reset not clearing `valid`), and each one made it
report FAIL.

## Project layout

```
app/        main.py, dataset.py, preprocessing.py, inspection.py,
            postprocessing.py, benchmark.py, reporting.py
inference/  base.py, cpu_backend.py, ai_suite_backend.py (placeholder), __init__.py
rtl/        image_stream.v, tb_image_stream.v
scripts/    download_dataset.py, build_memory_bank.py, calibrate_threshold.py
tests/      the pytest tests
datasets/, models/, results/   contents are not stored in git
```

## Path to AI Suite (not implemented, not verified)

Right now nothing runs on the AI Suite or on an FPGA. `inference/ai_suite_backend.py` is just a
placeholder that raises an error, and there are no FPGA results. Below is what I think the next
steps would be. I have not tried any of them.

1. Find out if the model (the ResNet18 part plus the nearest-neighbor search) can be exported to
   a format the AI Suite accepts. The nearest-neighbor search might have to stay on the host CPU.
2. Convert the model with the AI Suite on a machine that has it installed. The exact steps
   depend on the AI Suite example and the hardware.
3. Fill in `load_model` and `predict` in `inference/ai_suite_backend.py` so they return the same
   `Prediction` as the CPU backend.
4. Add `ai_suite` to the `--backend` choices in `app/main.py`. The rest of the app should not
   need changes.
5. Pick the threshold again, since a converted or quantized model can give scores on a different
   scale.
6. Benchmark on the board, and only compare with the CPU numbers if both were measured under
   documented, comparable conditions.

The Verilog module is not part of this path yet.

## Not done yet

AI Suite integration, running on an FPGA, hardware benchmarks, checking the anomaly maps,
a smarter way of choosing the memory patches (greedy coreset), and checking it on Windows.
