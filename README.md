# Applying Transfer Learning to Image Super-Resolution

**University:** Aarhus University, Department of Computer Science  
**Date:** December 2025   
**Authors:** Andreas W. R. Madsen, Benno Kossatz, Md Sadik Hasan Khan 

## 📝 Abstract
This project explores the potential of combining Transfer Learning with Convolutional Neural Networks (CNNs) for the Image Super-Resolution (SR) task. 
We implemented a custom CNN based on an encoder-decoder architecture, utilizing a pre-trained VGG16 network as a feature extractor.
The model was trained on the T91 dataset and validated on Set5 and Set14.

Our final model achieves an average PSNR of **26.17 dB** on Set5 and **23.87 dB** on Set14 for 4x upscaling, demonstrating a clear improvement over our baseline implementation.

## 🏗️ Architecture

The model utilizes an **Encoder-Decoder** structure:

* **Encoder:** Based on the VGG16 architecture pre-trained on ImageNet. We freeze the early layers to act as a feature extractor, capturing details from low-resolution inputs.
* **Decoder:** Consists of transpose convolution layers to upscale extracted features and reconstruct the high-resolution image.
* **Loss Function:** We utilize a combination of pixel-wise Mean Squared Error (MSE) and **Perceptual Loss** (based on high-level VGG features) to improve texture and edge preservation.


### Optimizations
To improve convergence and performance, we implemented:
* **AdamW Optimizer** & **Weight Decay** to prevent overfitting.
* **Learning Rate Scheduler** (dynamic reduction upon validation loss plateau).
* **Perceptual Loss** ($L_{perceptual}$) to enhance visual quality (SSIM).

## 📊 Dataset
* **Training:** [T91 Dataset](http://www.urbandictionary.com/) (91 images, ~22,000 patches).
    * *Preprocessing:* Patches generated with a size of 32x32 and stride of 14.
* **Validation:** Set5 and Set14 datasets.

## 🚀 Experiments & Results

We evaluated the model using **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index Measure).

### Performance Comparison (4x Upscaling)

| Model | Set5 PSNR (dB) | Set5 SSIM | Set14 PSNR (dB) | Set14 SSIM |
| :--- | :--- | :--- | :--- | :--- |
| **Bicubic** | 26.69 | 0.789 | 24.24 | 0.683 |
| **VGGSR1 (Baseline)** | 24.18 | 0.689 | 22.65 | 0.631 |
| **VGGSR (Final)** | **26.17** | **0.766** | **23.87** | **0.651** |
| SRCNN [1] | 30.49 | 0.8628 | 27.50 | 0.7513 |
| HAT-L [8] | 33.30 | 0.9083 | 29.47 | 0.8015 |

*[Table Data Source: 193]*

### Visual Analysis
While our PSNR is lower than state-of-the-art transformers, the **VGGSR** model excels in edge reconstruction compared to Bicubic interpolation.
* **Bird Image:** Bicubic handles out-of-focus backgrounds well but lacks sharpness.
* **Butterfly Image:** VGGSR produces sharper black edge details compared to the blurry output of Bicubic interpolation.

## 🛠️ Usage

### Prerequisites
* Python 3.x
* PyTorch (with Torchvision) 
* Pillow (PIL) 
* Matplotlib
* Numpy

### Training
To train the model using the T91 dataset patches:
```bash
python train.py --epochs 200 --batch_size 16 --lr 0.001

## Setup
1. Create a new virtual environment
2. Install correct PyTorch version https://pytorch.org/get-started/locally/
    - (Latest Cuda Version 12.4: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124`)
3. Install other requirements using `pip install -r requirements.txt`
