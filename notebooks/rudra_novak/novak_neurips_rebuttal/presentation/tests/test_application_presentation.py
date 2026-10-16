from __future__ import annotations

import sys
from pathlib import Path

import pytest


PRESENTATION_DIR = Path(__file__).resolve().parents[1]
REBUTTAL_ROOT = PRESENTATION_DIR.parent
sys.path.insert(0, str(PRESENTATION_DIR))

from build_application_presentation import (  # noqa: E402
    PresentationBuilder,
    RectElement,
    SlideSpec,
    build_slide_specs,
    load_eeg_torus_confusion_summary,
    load_results,
    validate_claims,
)


@pytest.fixture(scope="module")
def results():
    loaded = load_results(REBUTTAL_ROOT)
    validate_claims(loaded)
    return loaded


def test_final_numerical_claims(results):
    assert round(results["eeg"]["torus"]["mean_f1"], 3) == 0.670
    assert round(results["eeg"]["all_band"]["mean_f1"], 3) == 0.696
    assert round(results["motor"]["torus"]["mean_f1"], 3) == 0.181
    assert round(results["motor"]["all_band"]["mean_f1"], 3) == 0.212
    assert results["counts"] == {
        "eeg_total_sessions": 24,
        "eeg_valid_sessions": 21,
        "motor_unique_lfps": 341,
        "motor_full_lfps": 237,
    }


def test_significance_claims(results):
    assert results["eeg"]["torus_vs_relevant_p_holm"] == pytest.approx(
        5.7220458984375e-06
    )
    assert results["eeg"]["torus_vs_psd_p_holm"] == pytest.approx(
        5.7220458984375e-06
    )
    assert results["motor"]["torus_vs_relevant_p_holm"] == pytest.approx(
        2.316701451135059e-15
    )
    assert results["motor"]["torus_vs_psd_p_holm"] == pytest.approx(
        2.971313176048524e-13
    )


def test_slide_inventory_and_language(results):
    slides = build_slide_specs(results)
    assert len(slides) == 10
    assert sum(slide.kind == "core" for slide in slides) == 10
    assert sum(slide.kind == "backup" for slide in slides) == 0
    assert [slide.title for slide in slides] == [
        "From torus geometry to behavior",
        "Sleep geometry",
        "Sleep-stage decoding",
        "EEG feature comparison",
        "Across-hour stability",
        "Radius-feature ablation",
        "Reaching task",
        "Reach-direction decoding",
        "Replicates by animal",
        "Complementary geometry",
    ]

    all_text = "\n".join(
        [
            value
            for slide in slides
            for value in (slide.title, slide.script, slide.takeaway, slide.transition)
        ]
        + [element.text for slide in slides for element in slide.text_elements]
    ).lower()
    assert "all-band power" in all_text
    assert "feature ablation" in all_text
    assert "cross-animal transfer" not in all_text


def test_sleep_confusion_summary_uses_all_valid_hours():
    summary = load_eeg_torus_confusion_summary(REBUTTAL_ROOT)

    assert summary["n_sessions"] == 21
    assert summary["mean"].shape == (3, 3)
    assert summary["std"].shape == (3, 3)
    assert summary["mean"].sum(axis=1) == pytest.approx([1.0, 1.0, 1.0])
    assert summary["mean"].diagonal() == pytest.approx(
        [0.6088041309, 0.8264686011, 0.5778627496]
    )


def test_scene_graph_rejects_out_of_bounds_elements():
    with pytest.raises(ValueError, match="outside slide bounds"):
        SlideSpec(
            title="Bad geometry",
            section="Test",
            kind="core",
            elements=(RectElement(x=13.0, y=1.0, w=1.0, h=1.0),),
            script="Script",
            takeaway="Takeaway",
            transition="Transition",
            challenge_response="Response",
        )


def test_builder_smoke_renders_pptx_and_previews(tmp_path, results):
    slides = build_slide_specs(results)[:2]
    builder = PresentationBuilder(slides)
    pptx_path = tmp_path / "smoke.pptx"
    preview_dir = tmp_path / "previews"

    builder.render_pptx(pptx_path)
    previews = builder.render_previews(preview_dir)

    assert pptx_path.exists() and pptx_path.stat().st_size > 10_000
    assert len(previews) == 2
    assert all(path.exists() and path.stat().st_size > 10_000 for path in previews)
