# Six-stage feature projections (native tau = 3 ms)

Monkey T, y070316009-12, channel 7. All 198 short-delay trials contribute six corrected 300-ms stages each. Each figure shows one point per trial-stage window: open faint points are the 948 training windows, and filled points are the 240 held-out windows from saved fold 0. The same split, stage colors, and training-only median imputation/scaling are used in every panel. The 11 unusable geometry fits are imputed, not dropped.

- Geometry: all 15 features: [PCA](geometry_all_15/pca.png) | [LDA](geometry_all_15/lda.png) | [UMAP](geometry_all_15/umap.png)
- Geometry: radii and width: [PCA](geometry_radii_width_3/pca.png) | [LDA](geometry_radii_width_3/lda.png) | [UMAP](geometry_radii_width_3/umap.png)
- Geometry: orientation: [PCA](geometry_orientation_9/pca.png) | [LDA](geometry_orientation_9/lda.png) | [UMAP](geometry_orientation_9/umap.png)
- Geometry: fit quality: [PCA](geometry_fit_quality_3/pca.png) | [LDA](geometry_fit_quality_3/lda.png) | [UMAP](geometry_fit_quality_3/umap.png)
- Betti: raw distance: [PCA](betti_raw_3/pca.png) | [LDA](betti_raw_3/lda.png) | [UMAP](betti_raw_3/umap.png)
- Betti: RMS-normalized distance: [PCA](betti_rms_normalized_3/pca.png) | [LDA](betti_rms_normalized_3/lda.png) | [UMAP](betti_rms_normalized_3/umap.png)
- Geometry + Betti: raw distance: [PCA](geometry_plus_betti_raw_18/pca.png) | [LDA](geometry_plus_betti_raw_18/lda.png) | [UMAP](geometry_plus_betti_raw_18/umap.png)
- Geometry + Betti: RMS-normalized: [PCA](geometry_plus_betti_rms_normalized_18/pca.png) | [LDA](geometry_plus_betti_rms_normalized_18/lda.png) | [UMAP](geometry_plus_betti_rms_normalized_18/umap.png)

PCA and UMAP do not use stage labels during fitting. LDA uses the six training-stage labels, so training separation is not independent evidence of decoding. Held-out points are transformed without refitting. The displayed axes are method- and feature-set-specific; do not compare coordinate distances across figures. Betti values are the saved exact distance-averaged counts, with raw and RMS-normalized distance conventions kept separate. No geometry, Ripser, or decoder was rerun. Coordinate CSVs and method details accompany each PNG.

[Qualitative interpretation](INTERPRETATION.md) | [All-panel inspection sheet](inspection/contact_sheet.png) | [Validation](validation.json)
