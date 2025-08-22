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
