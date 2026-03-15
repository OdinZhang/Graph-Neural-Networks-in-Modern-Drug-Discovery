# HINGRL Data

This repository stores only the data contract in version control.
Dataset files are copied in by `scripts/prepare_data.py`.

Required files per dataset:
- `B-Dataset/AllEmbedding_DeepWalk.txt`
- `B-Dataset/AllNodeAttribute.csv`
- `B-Dataset/DrDiNum.csv`
- `B-Dataset/DrPrNum.csv`
- `B-Dataset/DiPrNum.csv`
- `B-Dataset/drugName.csv`
- `B-Dataset/diseaseName.csv`
- `F-Dataset/AllEmbedding_DeepWalk.txt`
- `F-Dataset/AllNodeAttribute.csv`
- `F-Dataset/DrDiNum.csv`
- `F-Dataset/DrPrNum.csv`
- `F-Dataset/DiPrNum.csv`
- `F-Dataset/drugName.csv`
- `F-Dataset/diseaseName.csv`

Auto-populated source:
- `/root/autodl-tmp/test_for_MoE/HINGRL-main/data/*`

The repository does not rerun OpenNE by default. Existing `AllEmbedding_DeepWalk.txt`
files are treated as the canonical prepared embeddings for demo purposes.
