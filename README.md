# Image Enhancement Pipeline for UXO Analysis  

A Python script designed to enhance images of unexploded ordnance (UXO) through a configurable multi-stage pipeline. Optimizes visual quality for downstream processing tasks like object detection or classification.

## Features  

- **White Balance Correction** (`grayworld`, `percentile`, or `lab`)
- **Brightness/Saturation Adjustment**
- **Contrast Enhancement** (CLAHE or Local Contrast Enhancement)
- **Denoising** (Bilateral Filtering)
- **Tonemapping** (Spatial or LECARM)

## Pipeline Progression Results

<table>
  <tr>
    <th>Stage</th>
    <th>Result</th>
  </tr>
  <tr>
    <td><b>Original Image</b></td>
    <td><img src="assets/uxo1.jpg" width="720"></td>
  </tr>
  <tr>
    <td><b>White Balance (Grayworld)</b></td>
    <td><img src="assets/uxo1-wb_grayworld.jpg" width="720"></td>
  </tr>
  <tr>
    <td><b>+ Denoising & LCE</b></td>
    <td><img src="assets/uxo1-wb_grayworld_denoise-d5-sc50.0-ss50.0_lce-degree3-smoothing0.2.jpg" width="720"></td>
  </tr>
  <tr>
    <td><b>+ LECARM Tonemapping</b></td>
    <td><img src="assets/uxo1_b10.0_s1.25-wb_grayworld_denoise-d5-sc50.0-ss50.0_lce-degree3-smoothing0.2_lecarm-camerasigmoid-down1.0-scale0.8.jpg" width="720"></td>
  </tr>
</table>

**Configuration Used for Final Enhanced Image**:

```yaml  
brightness_factor: 10.0
saturation_factor: 1.25

white_balance:
  enabled: True
  algorithm: grayworld
  percentile: 99.0

denoising:
  enabled: True
  diameter: 5
  sigma_color: 50.0
  sigma_space: 50.0

local_contrast_enhancement:
  enabled: True
  degree: 3
  smoothing: 0.2

lecarm_tonemapping:
  enabled: True
  camera_model: sigmoid
  downsampling: 1.0
  scaling: 0.8
```

## Comparison of differing methods

<table style="width:100%; text-align: center;">
  <tr>
    <th style="text-align: center;">CLAHE</th>
    <th style="text-align: center;">Local Contrast Enhancement</th>
  </tr>
  <tr>
    <td><img src="assets/uxo1_b10.0_s1.25-wb_grayworld_clahe-kernel3-clip1.0_denoise-d5-sc50.0-ss50.0.jpg" alt="CLAHE" width="400"></td>
    <td><img src="assets/uxo1-wb_grayworld_denoise-d5-sc50.0-ss50.0_lce-degree3-smoothing0.2.jpg" alt="Local Contrast Enhancement" width="400"></td>
  </tr>
  <tr>
    <th style="text-align: center;">LECARM Tonemapping</th>
    <th style="text-align: center;">Spatial Tonemapping</th>
  </tr>
  <tr>
    <td><img src="assets/uxo1_b10.0_s1.25-wb_grayworld_denoise-d5-sc50.0-ss50.0_lce-degree3-smoothing0.2_lecarm-camerasigmoid-down1.0-scale0.8.jpg" alt="LECARM Tonemapping" width="400"></td>
    <td><img src="assets/uxo1_b10.0_s1.25-wb_grayworld_denoise-d5-sc50.0-ss50.0_lce-degree3-smoothing0.2_stm-smoothing0.2-mid_tone0.5-tonal_width0.5-areas_dark0.5-areas_bright0.5.jpg" alt="Spatial Tonemapping" width="400"></td>
  </tr>
</table>

Local (spatial) tonemapping offers computational efficiency and superior performance in high dynamic range (HDR) scenes containing extreme luminance variations (e.g., simultaneous bright highlights and dark shadows). However, its edge-preserving smoothing operations inherently:  
1. **Reduce micro-texture detail** through regional averaging  
2. **Amplify existing noise** in low-contrast areas due to local contrast boosting  

Global methods like LECARM better preserve original texture structures through whole-image dynamic range compression, but sacrifice localized contrast adaptation - making them less effective for scenes requiring simultaneous highlight/shadow recovery.  

**Practical Trade-off**:  
- *Choose spatial tonemapping* for HDR recovery in resource-constrained environments  
- *Prefer global methods* when texture fidelity is critical and lighting is relatively uniform  

## Pipeline Order Rationale  

Enhancements are applied in this sequence to minimize artifacts and dependencies:

1. **Denoising**
2. **White Balance**  
3. **Brightness Adjustment**  
4. **Local Contrast Enhancement**  
5. **CLAHE**
6. **Spatial Tonemapping**
7. **LECARM Tonemapping**  
8. **Saturation Adjustment**  

*Why this order?* Early denoising prevents noise amplification. Color corrections (white balance) precede luminance adjustments, while saturation is finalized last to compensate for desaturation from prior steps.

## References

- X. Guo, Y. Li, and H. Ling, "LIME: Low-Light Image Enhancement via Illumination Map Estimation," in *IEEE Transactions on Image Processing*, vol. 26, no. 2, pp. 982-993, Feb. 2017, doi: [10.1109/TIP.2016.2639450](https://doi.org/10.1109/TIP.2016.2639450).

- Z. Ying, G. Li, Y. Ren, R. Wang, and W. Wang, "A New Low-Light Image Enhancement Algorithm Using Camera Response Model," in *2017 IEEE International Conference on Computer Vision Workshops (ICCVW)*, Venice, Italy, 2017, pp. 3015-3022, doi: [10.1109/ICCVW.2017.356](https://doi.org/10.1109/ICCVW.2017.356).

- Y. Ren, Z. Ying, T. H. Li, and G. Li, "LECARM: Low-Light Image Enhancement Using the Camera Response Model," in *IEEE Transactions on Circuits and Systems for Video Technology*, vol. 29, no. 4, pp. 968-981, April 2019, doi: [10.1109/TCSVT.2018.2828141](https://doi.org/10.1109/TCSVT.2018.2828141).

- Vassilios Vonikakis and Stefan Winkler, "A Center-Surround Framework for Spatial Image Processing," in *Proc. IS&T Int'l. Symp. on Electronic Imaging: Retinex at 50*, 2016, doi: [10.2352/ISSN.2470-1173.2016.6.RETINEX-020](https://doi.org/10.2352/ISSN.2470-1173.2016.6.RETINEX-020).

- V. Vonikakis and I. Andreadis, "Multi-Scale Image Contrast Enhancement," in *2008 10th International Conference on Control, Automation, Robotics and Vision*, Hanoi, Vietnam, pp. 856-861, Dec. 2008, doi: [10.1109/ICARCV.2008.4795629](https://doi.org/10.1109/ICARCV.2008.4795629).

- [Vassilios Vonikakis, Arapakis, and Andreadis, "Combining Gray-World Assumption, White-Point Correction, and Power Transformation for Automatic White Balance," 2011.](https://www.researchgate.net/publication/235350557_Combining_Gray-World_assumption_White-Point_correction_and_power_transformation_for_automatic_white_balance)

- G. Bianco, M. Muzzupappa, F. Bruno, R. Garcia, and L. Neumann, "A New Color Correction Method for Underwater Imaging," in *The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences*, vol. XL-5/W5, pp. 25-32, 2015, doi: [10.5194/isprsarchives-XL-5-W5-25-2015](https://doi.org/10.5194/isprsarchives-XL-5-W5-25-2015)
