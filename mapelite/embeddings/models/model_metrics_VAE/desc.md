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