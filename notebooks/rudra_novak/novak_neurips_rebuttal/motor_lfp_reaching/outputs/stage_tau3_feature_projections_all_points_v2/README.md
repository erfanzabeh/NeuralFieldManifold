# Six-stage feature projections: all points

Monkey T, y070316009-12, channel 7; native tau = 3 ms. Each panel contains all 1,188 corrected 300-ms trial-stage windows (198 per stage), using one identical point style and color only for stage. There is no train/test marker distinction in these descriptive views. Median imputation and standardization are fitted on all windows; 11 unusable geometry fits are imputed, not dropped.

- Geometry: all 15 features: [PCA](geometry_all_15/pca.png) | [LDA](geometry_all_15/lda.png) | [UMAP](geometry_all_15/umap.png)
- Geometry: radii and width: [PCA](geometry_radii_width_3/pca.png) | [LDA](geometry_radii_width_3/lda.png) | [UMAP](geometry_radii_width_3/umap.png)
- Geometry: orientation: [PCA](geometry_orientation_9/pca.png) | [LDA](geometry_orientation_9/lda.png) | [UMAP](geometry_orientation_9/umap.png)
- Geometry: fit quality: [PCA](geometry_fit_quality_3/pca.png) | [LDA](geometry_fit_quality_3/lda.png) | [UMAP](geometry_fit_quality_3/umap.png)
- Betti: raw distance: [PCA](betti_raw_3/pca.png) | [LDA](betti_raw_3/lda.png) | [UMAP](betti_raw_3/umap.png)
- Betti: RMS-normalized distance: [PCA](betti_rms_normalized_3/pca.png) | [LDA](betti_rms_normalized_3/lda.png) | [UMAP](betti_rms_normalized_3/umap.png)
- Geometry + Betti: raw distance: [PCA](geometry_plus_betti_raw_18/pca.png) | [LDA](geometry_plus_betti_raw_18/lda.png) | [UMAP](geometry_plus_betti_raw_18/umap.png)
- Geometry + Betti: RMS-normalized: [PCA](geometry_plus_betti_rms_normalized_18/pca.png) | [LDA](geometry_plus_betti_rms_normalized_18/lda.png) | [UMAP](geometry_plus_betti_rms_normalized_18/umap.png)

PCA and UMAP use feature values only. LDA uses stage labels to fit its axes and is explicitly supervised; apparent stage separation is not held-out decoding evidence. These are descriptive projections, not a new decoder. Axes and coordinate distances are specific to each plot. Betti values are the saved distance-averaged counts, with raw and RMS-normalized conventions kept separate. No geometry, Ripser, or decoder was rerun.

[All-panel inspection sheet](inspection/contact_sheet.png) | [Validation](validation.json)
