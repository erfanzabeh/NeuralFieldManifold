"""Small manual regression suite against frozen tutorial calculations.

Run explicitly with: python -m unittest discover -s tests -v
No workflow, import hook, or documentation build invokes this suite.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import neuralfieldmanifold as nfm

FIXTURES = Path(__file__).parent / "fixtures"


class TutorialParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def check_recording(self, name, sample, high, order, fraction, refresh, tau):
        recording = nfm.datasets.load(sample)
        with np.load(FIXTURES / f"{name}.npz") as reference:
            np.testing.assert_array_equal(recording.signal, reference["raw"])
            config = nfm.preprocessing.PreprocessConfig(
                notch_hz=50,
                band_hz=(1, high),
                output_rate_hz=500 if recording.sampling_rate_hz > 500 else None,
                envelope_band_hz=(1, 80),
                envelope_lowpass_hz=3,
            )
            processed = nfm.preprocessing.process(
                recording.signal,
                sampling_rate_hz=recording.sampling_rate_hz,
                config=config,
            )
            np.testing.assert_array_equal(processed.signal, reference["processed"])
            np.testing.assert_array_equal(processed.bandpassed, reference["bandpassed"])
            count = int(fraction * (len(processed.signal) - order))
            fit = nfm.models.AR(order=order).fit(processed.signal[: count + order])
            np.testing.assert_array_equal(
                np.r_[fit.intercept, fit.coefficients], reference["weights"]
            )
            prediction = fit.forecast(
                processed.signal,
                start_sample=order + count,
                n_samples=len(processed.signal) - order - count,
                refresh_samples=refresh,
            )
            np.testing.assert_array_equal(prediction, reference["predicted"])
            full = np.r_[
                processed.signal[:order],
                fit.forecast(
                    processed.signal,
                    start_sample=order,
                    n_samples=len(processed.signal) - order,
                    refresh_samples=refresh,
                ),
            ]
            np.testing.assert_array_equal(full, reference["full_prediction"])
            measured = nfm.models.prediction_metrics(reference["observed"], prediction)
            np.testing.assert_array_equal(
                [measured.mse, measured.mae, measured.correlation], reference["metrics"]
            )
            cloud = nfm.embedding.delay(
                reference["observed"].astype(np.float32), dimension=3, tau_samples=tau
            )
            np.testing.assert_array_equal(cloud, reference["cloud"])
            n_fft = min(4096, len(processed.signal))
            spectrum = nfm.spectral.welch(
                processed.signal,
                sampling_rate_hz=500,
                n_fft=n_fft,
                overlap_samples=n_fft // 2,
                frequency_range_hz=(0.5, 200),
            )
            np.testing.assert_array_equal(
                spectrum.frequencies_hz, reference["frequencies"]
            )
            np.testing.assert_allclose(
                spectrum.density, reference["psd"], rtol=2e-12, atol=1e-18
            )

    def test_monkey_tutorial(self):
        self.check_recording("monkey", "monkey_lfp", 50, 4, 0.9, 5, 30)

    def test_mouse_tutorial(self):
        self.check_recording("mouse", "mouse_lfp", 50, 3, 0.1, 10, 10)

    def test_mouse_lfp_tutorial(self):
        self.check_recording("mouse_lfp", "mouse_lfp", 20, 3, 0.1, 10, 10)

    def test_geometry_fits(self):
        for name, function in [
            ("one_torus", nfm.geometry.fit_ring),
            ("two_torus", nfm.geometry.fit_torus),
        ]:
            with (
                self.subTest(name=name),
                np.load(FIXTURES / f"{name}.npz") as reference,
            ):
                fit = function(reference["points"])
                fields = [
                    *fit.major_radii,
                    fit.tube_radius,
                    fit.score,
                    fit.error,
                    fit.mse,
                    fit.r_squared,
                    fit.hole_quality,
                    fit.fraction_in_hole,
                ]
                np.testing.assert_allclose(
                    fields, reference["fields"], rtol=1e-12, atol=1e-12
                )
                for field in ("center", "normal", "u_axis", "v_axis"):
                    np.testing.assert_allclose(
                        getattr(fit, field), reference[field], rtol=1e-12, atol=1e-12
                    )
                self.assertEqual(fit.features.shape, (15,))

    def test_deeplagfield_forward_losses_gradients(self):
        with np.load(FIXTURES / "deeplagfield.npz") as reference:
            torch.manual_seed(141)
            model = nfm.models.DeepLagField(sequence_length=96, hidden_size=16)
            x = torch.from_numpy(reference["signal"])
            torch.manual_seed(143)
            output = model(x, sample_order=True)
            for field, key in [
                ("coefficients", "coefficients"),
                ("order_logits", "logits"),
                ("order", "orders"),
                ("prediction", "prediction"),
            ]:
                np.testing.assert_array_equal(
                    getattr(output, field).detach().numpy(), reference[key]
                )
            losses = nfm.training.compute_losses(
                output, x, orders=model.orders, true_order=torch.tensor([2, 4, 6])
            )
            terms = [
                losses[k].item()
                for k in (
                    "reconstruction",
                    "order_classification",
                    "energy",
                    "smoothness",
                    "order_penalty",
                )
            ]
            np.testing.assert_array_equal(terms, reference["losses"])
            losses["total"].backward()
            gradients = np.concatenate(
                [p.grad.numpy().ravel() for p in model.parameters()]
            )
            np.testing.assert_array_equal(gradients, reference["gradients"])

    def test_training_and_checkpoint(self):
        torch.manual_seed(17)
        model = nfm.models.DeepLagField(sequence_length=96, hidden_size=16)
        x = torch.randn(4, 96)
        orders = torch.tensor([2, 3, 4, 6])
        history = nfm.training.train(
            model,
            [(x, orders)],
            validation_loader=[(x, orders)],
            config=nfm.training.TrainingConfig(epochs=1),
        )
        self.assertTrue(np.isfinite(history[0]["train"]["total"]))
        first, second = model(x), model(x)
        torch.testing.assert_close(first.prediction, second.prediction, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            model.save(path)
            restored = nfm.models.DeepLagField.load(path)
            torch.testing.assert_close(
                first.prediction, restored(x).prediction, rtol=0, atol=0
            )

    def test_explicit_input_checks(self):
        with self.assertRaises(ValueError):
            nfm.embedding.delay(np.zeros((3, 2)), dimension=3, tau_samples=1)
        with self.assertRaises(ValueError):
            nfm.embedding.delay(np.ones(2), dimension=3, tau_samples=1)
        with self.assertRaises(ValueError):
            nfm.preprocessing.process(
                np.ones(100),
                sampling_rate_hz=500,
                config=nfm.preprocessing.PreprocessConfig(output_rate_hz=300),
            )
        with self.assertRaises(ValueError):
            nfm.geometry.fit_torus(np.zeros((10, 3)))


if __name__ == "__main__":
    unittest.main()
