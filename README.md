# AI4SKIN leaderboard

[AI4SKIN leaderboard: Benchmarking Histopathology Foundation Models in a Multi-center Dataset for Skin Cancer Subtyping](https://doi.org/10.1007/978-3-031-98688-8_2)

[Pablo Meseguer<sup>1</sup>](https://scholar.google.es/citations?user=4r9lgdAAAAAJ&hl=es&oi=ao), [Rocío del Amor<sup>1</sup>](https://scholar.google.es/citations?user=CPCZPNkAAAAJ&hl=es&oi=ao), [Valery Naranjo<sup>1</sup>](https://scholar.google.com/citations?user=jk4XsG0AAAAJ&hl=es&oi=ao)

<sup>1</sup>[Universitat Politècnica de València (UPV)](https://www.upv.es/)

## AI4SKIN leaderboard

Classification performance in terms of balanced accuracy (BACC) for each combination of foundation model and multiple instance learning classifier. We also provide the Foundation Model - Silhouette Index (FM-SI) to measure model robustness against distribution shifts.  

| Foundation Model | ABMIL [2] | MI-SimpleShot [3] | FM-SI [ours] |
|------------------|-----------|-------------------|--------------|
| CONCH [FM1]      | 85.17%    | 72.37%            | 0.0934       |


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

The proposed benchmark for benchamrking foundation models is based on the AI4SKIN dataset. The dataset includes whole slide images (WSI) of cutaneous spindle cell neoplasms (CSC) for skin cancer subtyping. Dataset details are provided in the [Scientific Data manuscript](https://doi.org/10.1038/s41597-025-05108-3) and WSIs are publicly available in [Figshare](https://doi.org/10.6084/m9.figshare.27118035) [1].

From [this link](https://upvedues-my.sharepoint.com/:f:/g/personal/pabmees_upv_edu_es/EnVgZJtckMdJoPvDnqd3REUB_Oany7p6zFlQIwm3MQBLow?e=Mr8Sfg), you can manually download the files including the embeddings extracted with the corresponding foundation model. Each `.npy` file contain Nxd matrix where N denotes the number of patches in the slide and L the dimension of the instance-level features. We include at the moment the embeddings extracted with CONCH [FM1] and we plan to extend it to the other FM soon. 

* Weakly supervised classification 

Run weakly supervised classification based on multiple instance learning (MIL) for skin cancer subtyping. We implement attention-based MIL [2] and MI-SimpleShot [3]. In a near future, we are planning to extend the benchmarking to other MIL models. 

```
python main.py --folder <folder> --encoder CONCH --scenario <ABMIL/MISimpleShot>
```

* FM-SI: Foundation Model - Silhouette Index

We proposed the Foundation Model - Silhouette Index (FM-SI) to assess the model robustness against distribution shifts in terms of digitization scanner. FM-SI is based on t-SNE [4] dimensionality reduction and silhouette coefficients [5]. It measures FM robustess at the slide-level without requiring class labels.

To plot 2D t-SNE and get the FM-SI, you just need to add the corresponding flag to the execution. Script will not run the classification. 

```
python main.py --folder <folder> --encoder CONCH --get-fmsi
```

### To-do list

- [ ] Share embeddings of other foundation models
- [ ] Provide comparison of FM-SI with Robustness Index (RI) [6]
- [ ] Implement other MIL models (TransMIL [7])


### References

[1a] Del Amor, R., López-Pérez, M., Meseguer, P., Morales, S., Terradez, L., Aneiros-Fernandez, J., ... & Naranjo, V. (2025). A fusocelular skin dataset with whole slide images for deep learning models. Scientific Data, 12(1), 788.

[1b] Meseguer, Pablo; Terrádez, Liria; del Amor, Rocío; Naranjo, Valery; López-Pérez, Miguel; Aneiros-Fernández, José; et al. (2025). A Fusocelular Skin Dataset with Whole Slide Images for Deep Learning Models. figshare.

[2] Ilse, M., Tomczak, J., & Welling, M. (2018, July). Attention-based deep multiple instance learning. In International conference on machine learning (pp. 2127-2136). PMLR.

[3] Chen, R. J., Ding, T., Lu, M. Y., Williamson, D. F., Jaume, G., Song, A. H., ... & Mahmood, F. (2024). Towards a general-purpose foundation model for computational pathology. Nature medicine, 30(3), 850-862.

[4] Maaten, L. V. D., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of machine learning research, 9(Nov), 2579-2605.

[5] Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of computational and applied mathematics, 20, 53-65.

[6] de Jong, E. D., Marcus, E., & Teuwen, J. (2025). Current pathology foundation models are unrobust to medical center differences. arXiv preprint arXiv:2501.18055.

[7] Shao, Z., Bian, H., Chen, Y., Wang, Y., Zhang, J., & Ji, X. (2021). Transmil: Transformer based correlated multiple instance learning for whole slide image classification. Advances in neural information processing systems, 34, 2136-2147.

[FM1] Lu, M. Y., Chen, B., Williamson, D. F., Chen, R. J., Liang, I., Ding, T., ... & Mahmood, F. (2024). A visual-language foundation model for computational pathology. Nature medicine, 30(3), 863-874.