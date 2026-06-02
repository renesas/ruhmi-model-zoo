"""
Download a pretrained ResNet8 .h5 model from MLCommons Tiny and convert it
to TFLite (FP32 or INT8) using the CIFAR-10 test batch for representative
data when doing INT8 calibration.

Usage examples:
  # Download .h5 automatically and convert to INT8 (default)
  python download_model.py

  # Convert to FP32 only
  python download_model.py --mode fp32 --out CIFAR10ResNet8_fp32.tflite

  # Provide an existing .h5 instead of downloading
  python download_model.py --mode fp32 --h5 /path/to/pretrainedResnet.h5 --out CIFAR10ResNet8_fp32.tflite

  # INT8 conversion (calib_dir defaults to cifar-10-batches-py next to this script)
  python download_model.py --mode int8 --out CIFAR10ResNet8_int8.tflite --calib-dir /path/to/cifar-10-batches-py

  # Generate both FP32 and INT8 models
  python download_model.py --mode all
"""
import argparse
import os
import sys
import urllib.request
import numpy as np
import pickle
import tqdm
try:
    import tensorflow as tf
except Exception as e:
    print("TensorFlow is required to run this script. Install it in your environment.")
    raise

# Ensure local FP32_inference package files are importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# URL for the pretrained ResNet8 .h5 from MLCommons Tiny
H5_URL = (
    "https://raw.githubusercontent.com/mlcommons/tiny/master/"
    "benchmark/training/image_classification/trained_models/pretrainedResnet.h5"
)
DEFAULT_H5_PATH = os.path.join(HERE, "model", "pretrainedResnet.h5")


def download_h5(dest_path: str = None, force: bool = False) -> str:
    """Download the pretrained ResNet8 .h5 model if it doesn't already exist.

    Parameters
    ----------
    dest_path : str, optional
        Where to save the file.  Defaults to ``model/pretrainedResnet.h5``
        relative to this script.
    force : bool
        If True, re-download even when the file already exists.

    Returns
    -------
    str
        Absolute path to the downloaded .h5 file.
    """
    if dest_path is None:
        dest_path = DEFAULT_H5_PATH

    dest_path = os.path.abspath(dest_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.isfile(dest_path) and not force:
        print(f"Model already exists at: {dest_path} (use --force to re-download)")
        return dest_path

    print(f"Downloading pretrained ResNet8 .h5 from:\n  {H5_URL}")
    print(f"Saving to: {dest_path}")

    # Stream download with progress
    with urllib.request.urlopen(H5_URL) as response:
        total = int(response.headers.get("Content-Length", 0))
        with open(dest_path, "wb") as out_file:
            downloaded = 0
            block_size = 8192
            with tqdm.tqdm(total=total, unit="B", unit_scale=True, desc="Download") as pbar:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    pbar.update(len(chunk))

    print(f"Download complete: {dest_path} ({os.path.getsize(dest_path)} bytes)")
    return dest_path


def unpickle(file):
    """Load pickle file"""
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def load_test_batch(data_dir='cifar-10-batches-py'):
    """Load CIFAR-10 test batch (10,000 images)"""
    
    test_dict = unpickle(f'{data_dir}/test_batch')
    
    test_data = test_dict[b'data']
    test_labels = np.array(test_dict[b'labels'])  # Convert to numpy array
    test_filenames = test_dict[b'filenames']
    
    # Reshape from (10000, 3072) to (10000, 32, 32, 3)
    test_data = test_data.reshape((len(test_data), 3, 32, 32))
    test_data = np.rollaxis(test_data, 1, 4)  # Move channels to last
    
    return test_data.astype(np.float32), test_labels, test_filenames


def representative_data_gen(data_dir, num_samples=100):
    """Yield representative samples for TFLite converter.

    Each yielded element must be a list of input arrays (matching the model input).
    We produce float32 arrays in range [0,1] shaped (1,32,32,3) to match the inference
    preprocessing in `inference.py` (which normalises images by dividing by 255.0).
    """
    test_data, _, _ = load_test_batch(data_dir)
    count = min(num_samples, len(test_data))
    # Use tqdm progress bar to show calibration progress
    for i in tqdm.tqdm(range(count), desc='Calibration', unit='samples'):
        arr = test_data[i]  # shape [32,32,3], dtype float32 with values in 0..255
        inp = np.expand_dims(arr, axis=0).astype(np.float32)
        yield [inp]


def convert(h5_path: str, out_path: str, mode: str, calib_dir: str = None, num_calib: int = 100):
    print(f"Loading Keras model from: {h5_path}")
    model = tf.keras.models.load_model(h5_path, compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if mode == 'fp32':
        print("Converting to FP32 TFLite (no quantization)")
        converter.optimizations = []
        tflite_model = converter.convert()
    else:
        print("Converting to INT8 (full integer) with representative dataset")
        if calib_dir is None:
            # default to the cifar-10-batches-py folder next to this script
            calib_dir = os.path.join(HERE, 'cifar-10-batches-py')

        if not os.path.isdir(calib_dir):
            raise FileNotFoundError(f"Calibration data directory not found: {calib_dir}")

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_data_gen(calib_dir, num_samples=num_calib)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Ensure input/output are int8 for full integer quantization
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        
    with open(out_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Wrote TFLite model to: {out_path}")


def main():
    p = argparse.ArgumentParser(description='Download pretrained ResNet8 .h5 and convert to TFLite (fp32, int8, or both)')
    p.add_argument('--mode', choices=['fp32', 'int8', 'both', 'FP32', 'INT8', 'BOTH'], default='both')
    p.add_argument('--h5', default=None,
                   help='Path to Keras .h5 model file. If omitted the model is downloaded automatically.')
    p.add_argument('--out', default=None, help='Path to output tflite file (ignored when mode=all)')
    p.add_argument('--calib-dir', default=None, help='Path to CIFAR-10 batch folder for calibration')
    p.add_argument('--calib-num', type=int, default=1000, help='Number of calibration samples to use')
    p.add_argument('--force', action='store_true',
                   help='Force re-download of the .h5 model even if it already exists')

    args = p.parse_args()

    # --- Resolve h5 path (download if needed) ---
    h5_path = args.h5 if args.h5 else DEFAULT_H5_PATH

    if not os.path.isfile(h5_path) or args.force:
        h5_path = download_h5(dest_path=h5_path, force=args.force)
    else:
        print(f"Using existing .h5 model: {h5_path}")

    # --- Determine which modes to run ---
    if args.mode == 'both':
        modes = ['fp32', 'int8']
    else:
        modes = [args.mode]

    os.makedirs('model', exist_ok=True)

    for mode in modes:
        if args.out and len(modes) == 1:
            out_path = args.out
        else:
            base = f'Resnet_{mode}.tflite'
            out_path = os.path.join('model', base)

        convert(h5_path, out_path, mode, calib_dir=args.calib_dir, num_calib=args.calib_num)


if __name__ == '__main__':
    main()
