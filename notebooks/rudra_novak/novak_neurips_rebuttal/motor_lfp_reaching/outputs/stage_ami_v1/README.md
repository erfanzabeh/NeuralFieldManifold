# Independent stage AMI diagnostic

- `ami_pooled.png`: primary pooled curve and suggested delay.
- `ami_pooled_detail.png`: zoomed pooled curve, 10-40 ms, showing the shallow first minimum.
- `ami_by_stage.png`: descriptive curves for all six corrected stages.
- `ami_by_stage_detail.png`: zoomed stage curves, 10-70 ms, with shared axes.
- `ami_bin_sensitivity.png`: estimator sensitivity, without retuning.
- `REPORT.md`: findings, fold choices, and limitations.
- `INTERPRETATION.md`: qualifications about the selected minimum and pooling.
- `tables/`: exact curves, selections, frozen window identities, and existing F1 context.

Analysis: `stage_ami_analysis.py`; plot-only: `plot_stage_ami.py` and `plot_stage_ami_detail.py`. All previous outputs remain unchanged.
