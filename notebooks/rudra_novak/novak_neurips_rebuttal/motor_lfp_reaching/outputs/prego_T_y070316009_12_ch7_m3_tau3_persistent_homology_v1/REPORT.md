# T, Channel 7: Persistent Homology

**Completed:** 198/198 short-delay trials, 33 per direction; no failures,
subsampling, or approximations. Each calculation used the saved 494-point,
three-dimensional cloud at tau = 3 ms. Ripser 0.6.15 computed full H0/H1/H2
persistence with Euclidean distances and coefficients in F2.

**Findings**
- Every trial has finite H1 intervals. The longest H1 lifetime has median
  **0.623** (IQR **0.475-0.848**) in the existing normalized signal's distance units.
- After dividing by the cloud's centered RMS radius, the median is **0.364**
  (IQR **0.292-0.477**). Direction-specific medians range from **0.309 to 0.403**.
  Lifetime and radius distributions overlap substantially across directions;
  these plots do not establish direction-specific topology.
- H2 intervals occur in 197 trials and are generally short: the pooled interval
  lifetime median is **0.00645**. They are retained, not filtered away. Without
  null calibration, these intervals are not evidence of a higher-dimensional torus.
- Radius panels reuse **197 valid fits**. Original trial 128 is excluded only
  from radii because its earlier fit did not converge; its PH calculation succeeded.

**Deliverables:** 19 separate 600-dpi PNGs: two lifetime distributions, three
Betti-curve panels, two radius distributions, six geometry examples, and six
persistence diagrams. Dots are trials; distribution markers show median/IQR.
Betti lines show means with interquartile bands, not confidence intervals.

**Validation:** all saved intervals and Betti values independently validated;
all **25,822 protected files** remain unchanged. No geometric refitting, decoding,
significance testing, or parameter retuning occurred. These are descriptive
results from one selected recording, not cross-recording or cross-animal evidence.

The existing annular-band model and decoding results remain untouched.
