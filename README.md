# iWaveV3 - Advanced Neural Image Compression

A state-of-the-art deep learning-based image compression framework that combines wavelet-like transforms with entropy modeling for efficient and high-quality image compression.

## Features

- **Dual Model Architecture**: 
  - `iWaveV3_Obj`: Optimized for objective metrics (PSNR)
  - `iWaveV3_Perp`: Enhanced with perceptual processing for better visual quality

- **Advanced Techniques**:
  - Multi-level wavelet-like transforms
  - Adaptive quantization with subband importance weighting
  - Channel-aware quantization (YCbCr space)
  - Improved entropy modeling with mixture distributions
  - Perceptual post-processing

- **Progressive Training**: Adaptive compression levels from easy to hard

##  Performance Metrics

- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **BPP**: Bits Per Pixel (compression efficiency)

##  Installation & Setup

```bash
# Required dependencies
pip install torch torchvision numpy pillow scikit-image matplotlib opencv-python requests
```

##  Usage

### Basic Compression

```python
# Initialize models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
iwave_obj = iWaveV3_Obj(levels=3, transform_type='affine').to(device)
iwave_perp = iWaveV3_Perp(levels=3, transform_type='affine').to(device)

# Load and compress image
image_tensor = load_and_preprocess_image("your_image.jpg")
compressed, quantized_subbands, rate = iwave_obj(image_tensor, training=False)
```

### Training

```python
# Train on custom dataset
image_paths = glob.glob("/path/to/images/*")
dataset = ImageCompressionDataset(image_paths, image_size=256)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Progressive training
trained_model = progressive_training(model, dataloader, device, "iWaveV3-Obj")
```

### Testing

```python
# Test with different compression levels
results = test_single_image_clean(iwave_obj, iwave_perp, device)
```

##  Project Structure

```
iwavev3/
├── models/
│   ├── iWaveV3_Base.py          # Base compression model
│   ├── iWaveV3_Obj.py           # Objective quality model
│   └── iWaveV3_Perp.py          # Perceptual quality model
├── components/
│   ├── WaveletLikeTransform.py  # Multi-level transform
│   ├── EntropyModel.py          # Improved entropy coding
│   └── ResidualBlocks.py        # Network components
├── utils/
│   ├── metrics.py               # PSNR, SSIM, BPP calculations
│   ├── image_processing.py      # Image loading and preprocessing
│   └── visualization.py         # Results display
└── config.py                    # Configuration parameters
```

##  Configuration

Key parameters in `Config` class:

```python
class Config:
    levels = 3                    # Wavelet decomposition levels
    hidden_channels = 96          # Network channels
    num_mixtures = 5              # Entropy model mixtures
    batch_size = 4
    learning_rate = 1e-4
    image_size = 256
    rate_weight = 5e-4           # Compression rate weight
    subband_importance = [1.0, 1.3, 1.3, 1.8]  # LL, LH, HL, HH
```

##  Compression Levels

The framework supports multiple compression levels:

- **High Quality**: `qstep = 0.015` (Best quality, lower compression)
- **Balanced**: `qstep = 0.025` (Good quality/compression balance)
- **High Compression**: `qstep = 0.035` (Higher compression, acceptable quality)

##  Output Visualization

The pipeline generates comprehensive visualizations:

1. **Original Image**
2. **iWaveV3-Obj Compressed**
3. **iWaveV3-Perp Compressed**
4. **iWaveV3-Obj Reconstructed**
5. **iWaveV3-Perp Reconstructed**

With detailed metrics for each stage.

##  Technical Details

### Wavelet-like Transform
- Multi-level decomposition (default: 3 levels)
- Affine/additive transform units
- Learnable analysis and synthesis

### Entropy Modeling
- Gaussian mixture models for probability estimation
- Context-adaptive coding
- Improved rate-distortion optimization

### Quantization
- Adaptive quantization steps per subband
- Channel-aware quantization (luminance vs chrominance)
- Training-aware noise injection

##  Results Interpretation

- **PSNR > 30 dB**: Excellent quality
- **PSNR 25-30 dB**: Good quality
- **SSIM > 0.9**: High structural similarity
- **BPP 0.1-0.5**: Good compression ratio

##  Performance Tips

1. **GPU Acceleration**: Uses CUDA when available
2. **Memory Efficient**: Configurable batch sizes and image sizes
3. **Progressive Training**: Automatically adjusts difficulty
4. **Adaptive Quantization**: Optimizes for different content types


##Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` or `image_size`
2. **Slow Training**: Enable CUDA, reduce model complexity
3. **Poor Compression**: Adjust `rate_weight` and quantization steps

### Getting Help

- Check the configuration parameters
- Verify image preprocessing
- Monitor training logs for anomalies

---

**Note**: This is a research implementation. For production use, consider optimization for specific deployment scenarios and thorough testing on target datasets.
