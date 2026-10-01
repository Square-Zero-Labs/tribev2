"""Offline feature-extraction checks using small, randomly initialized models."""

import importlib.util
import unittest


@unittest.skipUnless(
    all(
        importlib.util.find_spec(name) is not None
        for name in ("neuralset", "transformers", "moviepy")
    ),
    "Requires the notebook inference dependencies",
)
class FeatureCompatibilityTests(unittest.TestCase):
    def test_feature_pipeline(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        import torch
        from transformers import (
            VJEPA2Config,
            VJEPA2Model,
            VJEPA2VideoProcessor,
            LlamaConfig,
            LlamaModel,
            Wav2Vec2BertConfig,
            Wav2Vec2BertModel,
            Dinov2Config,
            Dinov2Model,
        )
        from neuralset.extractors.video import _HFVideoModel
        from moviepy import ImageClip, VideoFileClip

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix="vjepa2-") as tmp:
            root = Path(tmp)
            video = root / "thumbnail.mp4"
            with ImageClip(
                np.full((32, 32, 3), 100, dtype=np.uint8), duration=1
            ) as clip:
                clip.write_videofile(
                    str(video), fps=4, codec="libx264", audio=False, logger=None
                )
            with VideoFileClip(str(video)) as clip:
                frames = np.stack(list(clip.iter_frames()))
            assert frames.shape == (4, 32, 32, 3)
            print("MoviePy encode/decode passed", flush=True)
            config = VJEPA2Config(
                hidden_size=32,
                num_attention_heads=4,
                num_hidden_layers=2,
                crop_size=32,
                patch_size=16,
                frames_per_clip=4,
                pred_hidden_size=32,
                pred_num_attention_heads=4,
                pred_num_hidden_layers=1,
            )
            VJEPA2Model(config).save_pretrained(root)
            VJEPA2VideoProcessor(
                crop_size={"height": 32, "width": 32},
                size={"shortest_edge": 32},
                do_sample_frames=False,
            ).save_pretrained(root)
            wrapper = _HFVideoModel(str(root), num_frames=4)
            states = wrapper.predict_hidden_states(frames)
            assert torch.isfinite(states).all()
            print(
                "neuralset VJEPA2 reload/processing/forward passed",
                states.shape,
                flush=True,
            )
        for name, model, inputs in [
            (
                "Llama",
                LlamaModel(
                    LlamaConfig(
                        hidden_size=32,
                        intermediate_size=64,
                        num_hidden_layers=2,
                        num_attention_heads=4,
                        num_key_value_heads=4,
                        vocab_size=32,
                    )
                ),
                {"input_ids": torch.tensor([[1, 2, 3]])},
            ),
            (
                "Wav2Vec2Bert",
                Wav2Vec2BertModel(
                    Wav2Vec2BertConfig(
                        hidden_size=32,
                        intermediate_size=64,
                        num_hidden_layers=2,
                        num_attention_heads=4,
                        feature_projection_input_dim=16,
                        conv_depthwise_kernel_size=3,
                    )
                ),
                {"input_features": torch.randn(1, 16, 16)},
            ),
            (
                "Dinov2",
                Dinov2Model(
                    Dinov2Config(
                        hidden_size=32,
                        intermediate_size=64,
                        num_hidden_layers=2,
                        num_attention_heads=4,
                        image_size=32,
                        patch_size=16,
                    )
                ),
                {"pixel_values": torch.rand(1, 3, 32, 32)},
            ),
        ]:
            with torch.inference_mode():
                out = model.eval()(**inputs, output_hidden_states=True)
            assert len(out.hidden_states) == 3
            assert torch.isfinite(out.last_hidden_state).all()
            print(name, "forward/hidden states passed", flush=True)


if __name__ == "__main__":
    unittest.main()
