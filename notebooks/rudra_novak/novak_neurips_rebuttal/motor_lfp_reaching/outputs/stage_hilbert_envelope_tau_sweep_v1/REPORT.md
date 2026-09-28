# Hilbert-envelope stage decoding

198 short-delay trials contributed six corrected 300-ms windows each. Geometry used m=3, K=1, and the unchanged 15-feature LDA pipeline.

The exploratory F1-selected delay was **21 ms**, with macro-averaged F1 **0.3318**.
The pooled AMI first-minimum delay was **23 ms**.
At the AMI delay, geometric F1 was **0.2999**.
New envelope-normalized tau=3 F1: **0.2725**; previous MAD tau=3: **0.2928**.

Geometry's conditional 95% intervals were 0.308-0.354 at 21 ms and 0.276-0.323 at 23 ms. AMI also selected 23 ms in each outer training fold and across the tested histogram-bin settings.

At 21/23 ms, adding geometry increased relevant-band F1 from 0.291 to 0.357/0.356 and average-PSD F1 from 0.194 to 0.347/0.333. All-band power alone scored 0.400; adding geometry scored 0.396/0.407. Both paired all-band improvement intervals included zero.

Each selected delay had 1,090 usable windows and 98 unusable windows: 86 nonconverged fits and 12 windows from normalization-failed trials 43 and 89. No normalization fallback was used. All trials remained in the main analysis with training-fold imputation. Complete-trial sensitivity analyses retained 121 trials at each delay and scored 0.359/0.301, respectively; their retained subsets need not be identical.

Spectral baselines retain their original raw-signal definition. Intervals use 2,000 whole-trial bootstrap resamples stratified by reach direction. Both nominal fixed-delay permutation p-values were 0.001 with 1,000 permutations. The F1 winner, its interval, and its p-value are not corrected for delay selection or independently validated. AMI is a heuristic, not evidence of optimal topology.

Full-trial envelopes use temporal context outside the windows: this is offline, single-recording analysis. No previous outputs were modified. No Betti calculation, PCA, PINN, or geometric parameter tuning was performed.
