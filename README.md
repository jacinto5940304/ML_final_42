# Mosaic Removal and Resolution Enhancement

_In this project, we address the challenge of mosaic removal and resolution enhancement by combining object detection (YOLO) with four different deep learning models, namely SRGAN, AutoEncoder, Vision Transformer (ViT), and a pretrained Restormer. Our pipeline first applies YOLO to detect regions of interest in the image or video, and then uses OpenCV to create mosaic patterns. Subsequently, each model attempts to demosaic and enhance the resolution of the input. Experimental results on our dataset demonstrate that the pretrained Restormer and SRGAN achieve superior SSIM and PSNR scores compared to other baselines._

## 1. Introduction


本專案致力於針對偵測到的物件進行**馬賽克化**處理，並透過各種**超解析度 (Super-Resolution) 與去馬賽克 (Demosaicing)** 技術來復原高品質影像。  
此流程包括：

1. **目標偵測 (Object Detection)**：使用 **YOLO** 框出需要馬賽克化的區域。  
2. **馬賽克化 (Mosaic)**：利用 **OpenCV** 在指定 bounding box 上覆蓋馬賽克。  
3. **影像復原 (Demosaic / Super-Resolution)**：用多種深度學習模型還原或增強影像品質，包括：
   - **SRGAN**
   - **AutoEncoder**
   - **Vision Transformer (ViT)**
   - **Pretrained Restormer**

---


## 2. Dataset Preparation (YOLO + Mosaic)

1. **資料集標註**：  
   - 使用 **YOLO** 針對原始影像 (或影片) 進行物件偵測，取得 bounding box 座標。
2. **馬賽克生成**：  
   - 透過 **OpenCV** 對偵測到的區域加上馬賽克，形成需要去馬賽克 (Demosaic) 或進行超解析度的訓練/測試資料。
3. **資料夾結構** (範例)：

## 3. Models Overview

本專案使用了多種模型與方法來進行去馬賽克與超解析度：

1. **SRGAN**  
- 生成對抗網路 (GAN) 架構，適用於影像超解析度 (Super-Resolution)。
- ![image](https://github.com/user-attachments/assets/8e217d06-1230-4e99-8f8a-03a7d9ec5bfa)

2. **AutoEncoder**  
- 以自編碼器學習影像的特徵，嘗試從馬賽克還原原始影像。
3. **Vision Transformer (ViT)**  
- 以 Transformer 結構處理影像碎片 (patches)，用來做影像重建或增強。


4. **Pretrained Restormer**  
- 一個針對影像復原 (Restoration) 任務而設計的高效能模型，提供較佳的去噪與去馬賽克效果。


---

## 4. Directory Structure

- CNN & SRGAN
   - ![image](https://github.com/user-attachments/assets/47a1d579-5aa3-46fa-95b4-6b2516d4e8ce)
- VIT & Autoencoder & Restormer
   - ![image](https://github.com/user-attachments/assets/e4e2c750-d328-452a-8374-4e853622ff10)

  
---

## 5. Installation

### 5-1. **Clone 專案**  

   git clone [[https://github.com/[TODO]/Mosaic_Removal_Enhancement.git](https://github.com/jacinto5940304/ML_final_42.git) ]<br>
   cd Mosaic_Removal_Enhancement

### 5-2. **Run Model** 
   #### 5-2-1. **SRGAN、AutoEncoder、Vision Transformer** : 
      Just run the file in vscode
   #### 5-2-2. **Restormer** :
      Here's the Training instructions:
   ##### 5-2-2-1. **Training**
   1. To download GoPro training and testing data, run
   ```
   python download_data.py --data train-test
   ```
   
   2. Generate image patches from full-resolution training images of GoPro dataset
   ```
   python generate_patches_gopro.py 
   ```
   
   3. To train Restormer, run
   ```
   cd Restormer
   ./train.sh Motion_Deblurring/Options/Deblurring_Restormer.yml
   ```

   **Note:** The above training script uses 8 GPUs by default. To use any other number of GPUs, modify [Restormer/train.sh](../train.sh) and [Motion_Deblurring/Options/Deblurring_Restormer.yml](Options/Deblurring_Restormer.yml)

   ##### 5-2-2-2  **Evaluation**

   Download the pre-trained [model](https://drive.google.com/drive/folders/1czMyfRTQDX3j3ErByYeZ1PM4GVLbJeGK?usp=sharing) and place it in `./pretrained_models/`

   ###### Testing on GoPro dataset

   - Download GoPro testset, run
   ```
   python download_data.py --data test --dataset GoPro
   ```
   
   - Testing
   ```
   python test.py --dataset GoPro
   ```
   
   ###### Testing on HIDE dataset
   
   - Download HIDE testset, run
   ```
   python download_data.py --data test --dataset HIDE
   ```
   
   - Testing
   ```
   python test.py --dataset HIDE
   ```
   
   ###### Testing on RealBlur-J dataset
   
   - Download RealBlur-J testset, run
   ```
   python download_data.py --data test --dataset RealBlur_J
   ```
   
   - Testing
   ```
   python test.py --dataset RealBlur_J
   ```
   
   ###### Testing on RealBlur-R dataset
   
   - Download RealBlur-R testset, run
   ```
   python download_data.py --data test --dataset RealBlur_R
   ```
   
   - Testing
   ```
   python test.py --dataset RealBlur_R
   ```
   
   ###### To reproduce PSNR/SSIM scores of the paper (Table 2) on GoPro and HIDE datasets, run this MATLAB script
   
   ```
   evaluate_gopro_hide.m 
   ```
   
   ###### To reproduce PSNR/SSIM scores of the paper (Table 2) on RealBlur dataset, run
   
   ```
   evaluate_realblur.py 
   ```

   ##### 5-2-2-3 **Testing Instructions**
   ###### Training

   1. To download GoPro training and testing data, run
   ```
   python download_data.py --data train-test
   ```
   
   2. Generate image patches from full-resolution training images of GoPro dataset
   ```
   python generate_patches_gopro.py 
   ```
   
   3. To train Restormer, run
   ```
   cd Restormer
   ./train.sh Motion_Deblurring/Options/Deblurring_Restormer.yml
   ```
   
   **Note:** The above training script uses 8 GPUs by default. To use any other number of GPUs, modify [Restormer/train.sh](../train.sh) and [Motion_Deblurring/Options/Deblurring_Restormer.yml](Options/Deblurring_Restormer.yml)
   
   ## Evaluation
   
   Download the pre-trained [model](https://drive.google.com/drive/folders/1czMyfRTQDX3j3ErByYeZ1PM4GVLbJeGK?usp=sharing) and place it in `./pretrained_models/`
   
   #### Testing on GoPro dataset
   
   - Download GoPro testset, run
   ```
   python download_data.py --data test --dataset GoPro
   ```
   
   - Testing
   ```
   python test.py --dataset GoPro
   ```
   
   #### Testing on HIDE dataset
   
   - Download HIDE testset, run
   ```
   python download_data.py --data test --dataset HIDE
   ```
   
   - Testing
   ```
   python test.py --dataset HIDE
   ```
   
   #### Testing on RealBlur-J dataset
   
   - Download RealBlur-J testset, run
   ```
   python download_data.py --data test --dataset RealBlur_J
   ```
   
   - Testing
   ```
   python test.py --dataset RealBlur_J
   ```
   
   #### Testing on RealBlur-R dataset
   
   - Download RealBlur-R testset, run
   ```
   python download_data.py --data test --dataset RealBlur_R
   ```
   
   - Testing
   ```
   python test.py --dataset RealBlur_R
   ```
   
   #### To reproduce PSNR/SSIM scores of the paper (Table 2) on GoPro and HIDE datasets, run this MATLAB script
   
   ```
   evaluate_gopro_hide.m 
   ```
   
   #### To reproduce PSNR/SSIM scores of the paper (Table 2) on RealBlur dataset, run
   
   ```
   evaluate_realblur.py 
   ```

## 6. Usage 
### 6.1 產生馬賽克資料
1. **Yolo detect**
2. **OpenCV 加馬賽克**

### 6.2 模型訓練
1. SRGAN
2. Vision Transformer
3. AutoEncoder
4. Restormer

### 6.3 推論 / Demo

## Pre-Experiment Hypotheses

Prior to conducting the experiment, the following hypotheses were formulated based on the characteristics of the models and their known behavior in related tasks. These hypotheses may be incorrect and require experimental validation.

### 1. **Restormer**  
   - Hypothesis: Due to the advantages of Transformer architectures in capturing global features, Restormer is expected to perform best in mosaic removal and resolution enhancement, achieving the highest quantitative metrics (e.g., PSNR and SSIM).  
   - Limitation: It might struggle with fine texture details, especially in highly pixelated test images.

### 2. **AutoEncoder**  
   - Hypothesis: AutoEncoder, with its structured compression and reconstruction process, is predicted to excel in preserving structural similarity (SSIM). However, its PSNR might lag behind Restormer as AutoEncoders often have limitations in restoring intricate details.

### 3. **Vision Transformer**  
   - Hypothesis: Vision Transformer is anticipated to perform well in maintaining structural consistency due to its patch-based attention mechanism. However, its overall restoration performance and quantitative metrics might fall short of Restormer, and it could require significantly more computational resources than AutoEncoder.

### 4. **SRGAN**  
   - Hypothesis: Due to the adversarial training technique, SRGAN is expected to produce visually appealing results, particularly in recovering high-frequency details. However, its quantitative performance (PSNR and SSIM) is likely to be the lowest among the tested models.

### 5. **Challenges**  
   - Hypothesis: All models are expected to face difficulties when dealing with severely pixelated images, especially SRGAN, which relies heavily on detail recovery. Transformer-based models (e.g., Restormer and Vision Transformer) are anticipated to demand higher computational resources and longer training times compared to other architectures.

### 6. **Overall Performance and Application Scenarios**  
   - Hypothesis: Transformer-based models (Restormer and Vision Transformer) are expected to outperform other architectures in overall performance but at a higher computational cost. AutoEncoder is hypothesized to offer a balance between performance and efficiency, while SRGAN might be more suitable for applications prioritizing visual quality over quantitative metrics.

---

### 6.4 評估

## Model Performance Summary

### **1. Restormer**  
- **PSNR**: 30.02 (Highest)  
- **SSIM**: 0.7051  
- **Strengths**:  
  - Demonstrated superior quantitative performance, achieving the highest PSNR.  
  - Excelled in restoring image quality at a broader scale.  
- **Weaknesses**:  
  - Struggled with intricate features.  
  - Room for improvement in capturing structural similarity and finer texture details.  

---

### **2. AutoEncoder**  
- **PSNR**: 24.31  
- **SSIM**: 0.7512 (Highest)  
- **Strengths**:  
  - Effectively captured structural similarity, achieving the highest SSIM.  
  - Maintained good overall image quality.  
- **Weaknesses**:  
  - Lagged behind Restormer in terms of PSNR and overall restoration performance.  

---

### **3. Vision Transformer**  
- **PSNR**: 24.39  
- **SSIM**: 0.7434  
- **Strengths**:  
  - Preserved structural consistency effectively.  
  - Competitive performance in SSIM compared to AutoEncoder.  
- **Weaknesses**:  
  - Slightly lower quantitative performance (PSNR and SSIM) than Restormer and AutoEncoder.  
  - Patch-based attention mechanism might have contributed to these limitations.  

---

### **4. SRGAN**  
- **PSNR**: 19.441 (Lowest)  
- **SSIM**: 0.6122 (Lowest)  
- **Strengths**:  
  - Produced visually appealing results, particularly in recovering high-frequency details, due to its adversarial training technique.  
- **Weaknesses**:  
  - Performed the weakest among all models in quantitative metrics (PSNR and SSIM).  
  - Struggled significantly in quantitative restoration tasks.  

---


## 7. Results

### **Overview**  
This project successfully integrated **object detection using YOLOv8** with four advanced deep learning models to tackle **mosaic removal** and **resolution enhancement**.

---

### **Key Results**  
1. **Restormer**:  
   - Achieved the best quantitative results with a **PSNR of 30.02** and an **SSIM of 0.7051**.  

2. **AutoEncoder**:  
   - Followed Restormer, delivering a **balance between quality and efficiency** with a high **SSIM of 0.7512**.

3. **Vision Transformer**:  
   - Performed competitively, maintaining **structural consistency**, but slightly lagged behind AutoEncoder and Restormer in quantitative metrics.

4. **SRGAN**:  
   - Lagged in quantitative metrics but produced **visually appealing results**, particularly in recovering **high-frequency details**.

---

### **Key Insights**  
- The **choice of model** is critical for different applications:  
  - **Transformer-based architectures** like Restormer provide excellent performance but require higher computational resources.  
  - **AutoEncoder** balances quality and computational efficiency.  

---

### **Future Work**  
- Explore **hybrid methods**, such as **combining AutoEncoder and Restormer**, to further enhance performance and achieve a better trade-off between quality, efficiency, and computational cost.

## 8. License
MIT License

Copyright (c) 

Permission is hereby granted, free of charge, to any person obtaining a copy...
