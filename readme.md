### **Project Description**

The goal of this project was to develop a machine learning model to detect and classify objects that change significantly between two consecutive frames in a video. The model leverages pixel-wise differences between frames as input to focus on regions where changes occur. Please refer to this link to see the code: https://github.com/nathanbehailuz/cv-final-project


1. **Dataset**:

   - A total of 740 images from [ziyuan-linn's cleaned dataset](https://github.com/ziyuan-linn/cv_final_data), derived from [VIRAT](https://viratdata.org/) were used for training and validation.
   - Each frame has bounding box annotations for objects of interest and their respective class labels.
   - An 80/20 train-validation split was used for model evaluation.

2. **Architecture**:

   - Model: Faster R-CNN with a ResNet-50 backbone and Feature Pyramid Network (FPN).
   - The model was adapted to process pixel-difference images and predict bounding boxes, class labels, and confidence scores.

3. **Loss Functions**:

   - Classification Loss: Cross-entropy loss for predicting object classes.
   - Bounding Box Loss: Smooth L1 loss for refining bounding box coordinates.
   - Total Loss: Weighted sum of classification and bounding box losses.

4. **Training Process**:

   - Optimizer: Adam with a learning rate of 0.001.
   - Batch size: 2 images per batch.
   - Training duration: 10 epochs.
   - Total parameters: 41.5 million (including backbone and detection heads).
   - Early stopping criteria: Model performance on validation data.

---

#### **Training Performance**

- **Loss Convergence**:

  - Initial training loss: 10.98.
  - Final training loss after 10 epochs: 2.98.
  - Validation loss: 1.15.

- **Threshold Optimization**:

  - The confidence threshold for inference was dynamically learned using the validation set.
  - Optimal threshold: 0.50, achieving the best F1-score of 0.82.

- **Metrics**: 
  - Precision: 64%
  - Recall: 71%
