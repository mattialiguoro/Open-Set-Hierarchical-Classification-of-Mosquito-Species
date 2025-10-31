# Mosquito Vector Classification via Deep Neural Networks

## Overview  
Developed in collaboration with the *Istituto Superiore di Sanità (ISS)* (Italian National Institute of Health) and under the supervision of Prof. Stefano Giagu (*Sapienza University of Rome*), this project aims to support entomologists and public health authorities in the **automatic classification of disease vector mosquitoes via deep neural networks (DNNs)**.  

The system combines state-of-the-art deep learning techniques with **hierarchical and open-set classification**, providing robust and reliable genus- and species-level identification even under challenging conditions such as **imbalanced datasets, damaged specimens, and out-of-distribution inputs**.  

Leveraging a **ConvNeXtV2-Huge backbone**, a **multi-task architecture**, and a **modified Evidential Deep Learning (m-EDL)** approach for uncertainty quantification, the model achieves near state-of-the-art results while maintaining robustness and interpretability.  

Developed and fine-tuned on the **Leonardo HPC**, the 10th most powerful supercomputer in the world, this project provided a valuable opportunity to learn and utilize one of the most advanced computational platforms available, gaining expertise in **high-performance deep learning pipelines**.  

*(For the project article, check the “Report and presentation” folder.)*  


## Repository Structure  

```
├── Code/
│   ├── Train_and_inference.ipynb
│   ├── data_list_alberto.py
│   ├── utils/
│   └── ...
├── Report and presentation/
│   ├── Project_Report.pdf
│   └── Presentation.pdf
└── README.md
```

### Code Folder  
The **`Code`** folder contains all the files required for training and inference, except for the segmentation code, which is proprietary to the ISS and therefore not included in this repository. It was provided as-is and was not part of my work.  
Training can be performed by downloading the pre-trained model from **timm**.  
By default, the `pretrained` option is set to **False**, since I used **fine-tuned weights from ImageNet-1k** (loaded from my `$WORK` directory on Leonardo) before performing the final fine-tuning on my dataset.  

The **`utils`** folder and the **`data_list_alberto.py`** file are pre-existing scripts provided for data handling, as are the first few cells of the **`Train_and_inference.ipynb`** notebook.  

### Train_and_inference.ipynb  
This notebook contains the **core of the project**, including both the **training** and **inference** sections.  
It is the exact code used to produce the results presented in the final report and presentation.  

To run **inference**, simply:
1. Execute all cells up to (and including) the *model cell* in `Train_and_inference.ipynb`.
2. Load the pre-trained model available at this [Google Drive folder](https://drive.google.com/drive/folders/14rYu47K1gCN46UN7HOyw08LQ_Q3MHbPY?usp=sharing). This folder contains:
- The **best trained model**, fine-tuned on the custom dataset.  
- The **training history** of accuracy and loss metrics for both **gender** and **species** classification on training and validation sets.  


## Notes  
- Adjust **file paths** according to your own setup and directory structure before running the code.  
- Ensure all required dependencies are installed (see environment requirements if available).  


---

If you have any questions about the implementation, would like further clarifications, or have suggestions for improvement, feel free to reach out — I’ll be happy to discuss and help.

Mattia Liguoro
mattialiguoro17@gmail.com
