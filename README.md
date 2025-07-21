# AI4SKIN leaderboard

[AI4SKIN leaderboard: Benchmarking Histopathology Foundation Models in a Multi-center Dataset for Skin Cancer Subtyping](https://doi.org/10.1007/978-3-031-98688-8_2)

[Pablo Meseguer<sup>1</sup>](https://scholar.google.es/citations?user=4r9lgdAAAAAJ&hl=es&oi=ao), [Rocío del Amor<sup>1</sup>](https://scholar.google.es/citations?user=CPCZPNkAAAAJ&hl=es&oi=ao), [Valery Naranjo<sup>1</sup>](https://scholar.google.com/citations?user=jk4XsG0AAAAJ&hl=es&oi=ao)

<sup>1</sup>[Universitat Politècnica de València (UPV)](https://www.upv.es/)

## AI4SKIN leaderboard

In this work, we provide an extensive evaluation of histopathology foundation models in a benchmark for skin cancer subtyping on whole slide images (WSI) from the multi-center [AI4SKIN dataset](https://doi.org/10.1038/s41597-025-05108-3).

The table provides the classification performance in terms of balanced accuracy (BACC) for each combination of foundation model and multiple instance learning classifier. We also provide the Foundation Model - Silhouette Index (FM-SI) to measure model robustness against distribution shifts.


| Foundation Model                                    | [ABMIL](https://proceedings.mlr.press/v80/ilse18a.html) | [MI-SimpleShot](https://doi.org/10.1038/s41591-024-02857-3) | [FM-SI](https://doi.org/10.1007/978-3-031-98688-8_2) |
|-----------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------|------------------------------------------------------|
| [CONCH](https://doi.org/10.1038/s41591-024-02856-4) | 85.17%                                                  | 72.37%                                                      | 0.0934                                               |
| [UNI](https://doi.org/10.1038/s41591-024-02857-3)   | 82.51%                                                  | 68.28%                                                      | 0.5867                                               |



> **Note**  
> Results may vary from those presented in the paper as this repository uses the final version of the dataset containing 626 slides. We do not rely on the official partitions of the dataset as we used a patient-stratified 5-fold cross-validation. 

### Setting up AI4SKIN leaderboard

Clone repository and intall a compatible torch version with your GPU and required libraries.

```
git clone https://github.com/cvblab/AI4SkIN_leaderboard.git
cd AI4SkIN_leaderboard
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

## Usage

* Data downloading for reproducibility

The proposed benchmark for benchamrking foundation models is based on the AI4SKIN dataset. The dataset includes whole slide images (WSI) of cutaneous spindle cell neoplasms (CSC) for skin cancer subtyping. Dataset details are provided in the [Scientific Data manuscript](https://doi.org/10.1038/s41597-025-05108-3) and WSIs are publicly available in [Figshare](https://doi.org/10.6084/m9.figshare.27118035).

From [this link](https://upvedues-my.sharepoint.com/:f:/g/personal/pabmees_upv_edu_es/EnVgZJtckMdJoPvDnqd3REUB_Oany7p6zFlQIwm3MQBLow?e=Mr8Sfg), you can manually download the files including the embeddings extracted with the corresponding foundation model. Each `.npy` file contain Nxd matrix where N denotes the number of patches in the slide and L the dimension of the instance-level features. We include at the moment the embeddings extracted with CONCH and we plan to extend it to the other FM soon. 

* Weakly supervised classification 

Run weakly supervised classification based on multiple instance learning (MIL) for skin cancer subtyping. We implement attention-based MIL and MI-SimpleShot.
```
python main.py --folder <folder> --encoder <CONCH/UNI> --scenario <ABMIL/MISimpleShot>
```

* FM-SI: Foundation Model - Silhouette Index

We proposed the Foundation Model - Silhouette Index (FM-SI) to assess the model robustness against distribution shifts in terms of digitization scanner. FM-SI is based on t-SNE  dimensionality reduction and silhouette coefficients . It measures FM robustess at the slide-level without requiring class labels.

To plot 2D t-SNE and get the FM-SI, you just need to add the corresponding flag to the execution. Script will not run the classification. 

```
python main.py --folder <folder> --encoder <CONCH/UNI> --get-fmsi
```

### To-do list

- [ ] Share embeddings of other foundation models
- - [ ] Implement other MIL models (TransMIL)
- [ ] Provide comparison of FM-SI with Robustness Index (RI)