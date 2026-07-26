# Rebuttal calculations

This folder contains compact rebuttal units. Each unit reads existing NeuralFieldManifold code, notebooks, data, or guide references as source material and writes new work only inside its own folder.

## Scope rules

- Work only under `notebooks/rudra_novak/rudra_neurips_rebuttal/calculations/`.
- Treat `../guide/` as reference material. Do not edit copied notebooks, PDFs, docx files, or reference images there.
- Treat the rest of `/home/dev/NeuralFieldManifold` as a read-only dependency unless the user explicitly says otherwise.
- Each unit owns its own `cache/` and `plots/` folders.
- Reuse cached outputs when present. Recompute only when requested.
- Do not stage, commit, or push rebuttal work.

## Units

- `ref1_topology_assessment_regen/`: regenerate and improve the topology assessment components from `guide/ref1.png`.
- `ref2_sleep_decoding_regen/`: regenerate and improve the sleep decoding components from `guide/ref2.png`.
- `ref3_population_null_regen/`: regenerate and improve the population/null summary components from `guide/ref3.png`.
- `eunji_decode/`: recreate the ref2-style decoding analysis using Eunji's EKEZ LFP recordings and spreadsheet labels.

## Shared dependency pattern

The format template suggests creating a new project package. For this rebuttal, use the existing NeuralFieldManifold repo instead:

- Import or copy minimal helper logic only inside the active unit when needed.
- Prefer reading existing notebooks as provenance, then extracting small runnable scripts into the unit.
- Do not modify the copied guide notebooks.
- Do not move or rewrite original project data.
