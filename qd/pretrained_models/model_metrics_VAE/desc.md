## Description of models

### model_metrics_VAE_mixRng_tita_1.pth
- VAE model trained on a mixed (perlin/uniform) dataset of 20k track metrics.
- Trained on tracks created with the Tita driver, which has a more stable and consistent behavior than the previous driver, which should lead to better quality metrics and thus better embeddings.

### model_metrics_VAE_mixRng_tita_aug_1.pth
- VAE model trained on a mixed (perlin/uniform) dataset of 20k track metrics created with Tita driver.
- Adds data augmentation with rolling to try to make it more invariant.
### model_metrics_VAE_mixRng_tita_aug_2.pth
- VAE model trained on a mixed (perlin/uniform) dataset of 20k track metrics created with Tita driver, with augmentation with rolling.
- Adds per sample recon loss calculation.

### model_metrics_VAE_mixRng_tita_constrastive_1.pth
- VAE model trained on a mixed (perlin/uniform) dataset of 20k track metrics created with Tita driver.
- Adds contrastive loss to try to make rolled versions of the same track closer in the embedding space. 

### model_metrics_VAE_mixRng_tita_constrastive_2.pth
- VAE model trained on a mixed (perlin/uniform) dataset of 20k track metrics created with Tita driver.
- contrastive weight increased to 50

### model_metrics_VAE_mixRng_tita_constrastive_3.pth
- VAE model trained on a mixed (perlin/uniform) dataset of 20k track metrics created with Tita driver.
- uses infoNCE contrastive loss with temp of 0.05 and weight of 2

### model_metrics_VAE_mixRng_tita_constrastive_4.pth
- VAE model trained on a mixed (perlin/uniform) dataset of 20k track metrics created with Tita driver.
- uses infoNCE contrastive loss with temp of 0.03 and weight of 2, validation accuracy was 86.09%

### model_metrics_VAE_mixRng_tita_constrastive_5.pth
- VAE model trained on a mixed (perlin/uniform) dataset of 20k track metrics created with Tita driver.
- uses infoNCE contrastive loss with temp of 0.005 and weight of 2, validation accuracy was 94% but recon is higher and there are lots of collapsed dimensions.

### model_metrics_VAE_mixRng_tita_constrastive_6.pth
- VAE model trained on a mixed (perlin/uniform) dataset with winding canonicalization of 20k track metrics created with Tita driver.
- uses infoNCE contrastive loss with temp of 0.03 and weight of 2, uses the new dataset with winding canonicalization, so i removed the canonicalization from preprocessing.

### model_metrics_VAE_mixRng_tita_circular_1.pth
- VAE model trained on a mixed (perlin/uniform) dataset with winding canonicalization of 20k track metrics created with Tita driver.
- uses completely new architecture using circular convolutions.

### model_metrics_VAE_mixRng_tita_circular_2.pth
Changes from model_metrics_VAE_mixRng_tita_circular_1.pth:
- Adds lr scheduler in the training phase + weighted loss

### model_metrics_VAE_mixRng_tita_circular_3.pth
Changes from model_metrics_VAE_mixRng_tita_circular_2.pth:
- Uses new dataset with also smaller track size (between 1 and 10 instead of 4 and 10)

### model_metrics_VAE_mixRng_tita_circular_4.pth
kaggle name: "no max_beta check for EarlyStopping"
- Fixes issues with padding , adds positional encoding, uses ChannelLayerNorm instead of batchnorm, removes max_beta check from EarlyStopping

### model_metrics_VAE_mixRng_tita_circular_5.pth
kaggle name: "standard trackSize"
- uses standard track size (between 4 and 10) so the dataset20k_mixedRng_tita_winded.npz

### model_metrics_VAE_mixRng_tita_circular_6.pth
kaggle name: "n_layers=5"
- Complete revamp and refinement of the architecture, best performing model so far.

### model_metrics_VAE_mixRng_tita_circular_7.pth
- kaggle name: "without discarding k= 0 (version 2)"
- Changes the pooling strategy (normalized_dft_pooling) 

### model_metrics_VAE_mixRng_tita_circular_canon_1.pth
- VAE model trained on a mixed (perlin/uniform) dataset with new canonicalization of 20k track metrics created with Tita driver.
