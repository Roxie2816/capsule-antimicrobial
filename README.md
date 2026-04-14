# CapMolPred: Multi-Source Language Models Fusion with Dynamic Routing Capsule Network for Antimicrobial Compound Identification
We propose CapMolPred, which integrates five domain-adapted chemistry language models for high-dimensional small-molecule encoding and adopts cross-attention for heterogeneous embedding alignment. It also combines a capsule network with dynamic routing and a novel asymmetric loss function to achieve superior inhibitory potency prediction against Escherichia coli and Acinetobacter baumannii compared to conventional methods.
- **Access CapMolPred at**: [https://dmci.xmu.edu.cn/CapMolPred/indexpage.php](https://dmci.xmu.edu.cn/CapMolPred/indexpage.php)

- [Environment Setup](#install)
- [Quick Start](#quick-start)
- [User Interface](#user-interface)

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

### Data Partitioning & Training

To rigorously evaluate the model's performance and its generalization capability across unseen chemical spaces, we implemented a 5-fold scaffold split strategy:

- **Scaffold Split Logic:** Unlike traditional random splitting, scaffold splitting ensures that all molecules sharing the same core scaffold reside within the same set (either training or test).
- **Preventing Data Leakage:** This approach prevents the model from achieving artificially high performance by "memorizing" similar structural analogs, thereby ensuring a more robust evaluation of its ability to generalize to novel structures.

**Training Protocol:**
- **Ratio:** An approximate 8:2 training-to-test ratio was maintained.
- **Framework:** Developed using TensorFlow and Keras.
- **Hyperparameters:** Optimized using the Adam optimizer (initial LR=0.0001) with a batch size of 64.
- **Early Stopping:** Training halts if no improvement in validation accuracy is observed for 50 consecutive epochs to prevent overfitting.

---

## Quick Start

**Notice**  
Make sure you have 3 folder (data, model, output) before running the script.
args --model-name and --drug-descriptor are needed
- data/: Curated raw SMILES corpora employed for model training, hold-out testing, and prospective prediction.
- model/: Pre-trained feature extractors and associated checkpoints utilized in the molecular representation pipeline.
- output/: End-to-end antimicrobial-activity predictions, encompassing extracted molecular descriptors, serialized model weights, and comprehensive training/validation metrics exported as CSV artifacts.
  
**Feature extraction**  
```bash
python drugclassification.py --input-path data/sample.csv --model-name bert_chemmolefusion_capsule --drug-descriptor fusion
```
**Train and Test**  
```bash
python drugclassification.py  --input-path data/ecoli_predict.csv --model-name bert_chemmolefusion_capsule --drug-descripter fusion --train --batch-size 64 -e 100 -dp data -g 0 -sl 1024
```
**Prediction**  
```bash
python drugclassification.py  --input-path data/sample_predict.csv --model-name bert_chemmolefusion_capsule --drug-descripter fusion --predict
```

---

## User Interface
CapMolPred offers a user-friendly interface with the following key components:
1. **Input Section**: Submit sequences in SMILES format (manual entry or file upload).
3. **Results Visualization**:
   - **Table View**: Detailed tables of predicted activity and corresponding probabilities.
   - **Structure Displays**: 2D chemicals structure visualizations.
3. **Export Options**: Download results in tab-delimited or XML formats.
---


## Contact
For technical support or inquiries, please contact:  
Dr. Yixian Huang  
Address: Faculty of Computer Science and Control Engineering, Shenzhen University of Advanced Technology, Shenzhen 518107, China  
Email: huangyixian@suat-sz.edu.cn
