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
