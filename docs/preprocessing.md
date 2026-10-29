# Preprocessing is part of the trained model

**Use the supplied 640 × 640 PNGs to reproduce training and image-based inference directly.**
They already contain the preprocessing used by our experiments. You do not need the raw
I/Q recordings or a spectrogram renderer for this route.

**Changing preprocessing parameters or library versions changes the spectrogram pixels.**
The resulting input distribution can differ enough that the released pretrained detector
no longer works reliably, even when the image looks similar and has the correct size.
For your own recordings or a different rendering pipeline, **regenerate the complete dataset
consistently and retrain your own model**. Keep the same preprocessing at deployment.

## Reference raw-I/Q renderer

| Component | Setting |
|---|---|
| Signal input | complex64 I/Q; sample rate 30.72 MS/s |
| Spectrogram | SciPy `signal.spectrogram`, magnitude, two-sided; 512-point window, 384-sample overlap |
| Frequency / intensity | FFT shift; `10*log10(magnitude + 1e-10)` |
| Dynamic range | Per-frame 5th and 95th percentiles |
| Rendering | Matplotlib Agg, `viridis`, transposed spectrogram, upper origin, no axes/borders, 100 DPI |
| Model image | 640 × 640; RGB decoded from the rendered PNG; float tensor divided by 255 |

The tested renderer profiles differ between the PC and Jetson:

| Library | PC renderer | Jetson renderer |
|---|---|---|
| NumPy | 2.2.4 | 2.2.4 |
| SciPy | 1.14.1 | 1.17.1 |
| Matplotlib | 3.10.8 | 3.10.8 |
| Pillow | 11.2.1 | 11.2.1 |

FFT/window settings, overlap, logarithmic convention, percentile limits, colormap,
axis orientation, interpolation and image size all affect the input. The complete
implementation is in [sigmf_preprocessor.py](../cbrs/sigmf_preprocessor.py); omitted
SciPy/Matplotlib arguments inherit the defaults of the pinned versions.

Training and the CEDAR quality sweep use the PC environment; local speed baselines
and MDI use the [Jetson environment](3_yolo_mdi.md#environment-on-every-node).
General training/inference libraries may use compatible alternative versions.
**Do not apply that flexibility blindly to raw-I/Q rendering:** even a small change
to rendering versions or parameters can invalidate predictions from the pretrained
weights. Keep the chosen profile and settings consistent, and verify pixels before
using regenerated inputs. The supplied PNGs are the reference.

After downloading the official raw dataset, check rendering against one reference frame
from each split and LTE-bandwidth group:

```bash
python scripts/check_rendering.py --raw-source /path/to/IEEE_UPLOAD
```

All 12 selected frames were pixel-identical to the supplied PNGs in checks with
both SciPy 1.14.1 and 1.17.1. The 1.17.1 compatibility check ran on the PC with
temporary dependencies; it was not a new Jetson performance run. These checks
do not establish compatibility with arbitrary versions or settings. Run the pixel
check on your actual deployment environment.

The command records pixel differences in `experiment_reruns/rendering_check.json` and
fails on differences beyond the small rounding tolerance. This check does not alter
the PNGs or weights. Raw-I/Q local and MDI timing includes rendering; PNG-only tests
measure detection on already prepared images and cannot reproduce that preprocessing cost.

The same rule applies to INT8 calibration in CBRS-CEDAR-RT: use training images rendered
with the same settings as the inference images. Quantization does not correct a mismatch
between training and deployment preprocessing.
