# capsule-antimicrobial
Our model incorporates a unique capsule network architecture and introduces innovations in loss function selection and feature processing modules, demonstrating superior performance in predicting inhibitory activities against Escherichia coli and Acinetobacter baumannii. 

**Feature extraction**  
python drugclassification.py --input-path data/sample.csv --model-name bert_chemmolefusion_capsule --drug-descriptor fusion

**Train and Test**  
python drugclassification.py  --input-path data/ecoli_predict.csv --model-name bert_chemmolefusion_capsule --drug-descripter fusion --train --batch-size 64 -e 100 -dp data -g 0 -sl 1024

**Prediction**  
python drugclassification.py  --input-path data/sample_predict.csv --model-name bert_chemmolefusion_capsule --drug-descripter fusion --predict

**Notice**  
Make sure you have 3 folder (data, model, output) before running the script.

<!-- 目录 -->
- [Environment Setup](#install)
- [Quick Start](#quick-start)

<!-- 任意位置手动放锚点 -->
<a name="install"></a>
## Environment Setup
CapMolPred requires several dependencies. Here are the main requirements:

### Core Dependencies
- Python 3.10
- PyTorch 1.11.0 
- Tensorflow 2.20.0

### Key Python Packages
- numpy==1.26.4
- scipy==1.10.1
- pandas==2.3.1
- matplotlib==3.9.2
- tqdm==4.67.1

### Pip install Packages
- transformers==4.39.3
- pytorch-lightning==2.0.3
- pytorch-fast-transformers==0.4.0

### Installation
You can set up the environment using conda:
```bash
conda env create -f environment.yml
conda activate TBDMA
```
## Quick Start

写你的教程...
