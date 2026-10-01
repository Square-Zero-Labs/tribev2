"""Exercise native audio/video operators with the installed PyTorch stack."""
import importlib.util
import subprocess
import sys
import unittest


@unittest.skipUnless(
    all(importlib.util.find_spec(name) is not None for name in ("torch", "torchvision", "torchaudio")),
    "Requires PyTorch, TorchVision, and TorchAudio",
)
class TorchExtensionTests(unittest.TestCase):
    def test_native_extensions(self):
        subprocess.run(
            [sys.executable, "-c", """
import torch
import torchvision
import torchaudio
assert torch.__version__.split('+')[0] == torchaudio.__version__.split('+')[0]
assert torchaudio._extension._IS_TORCHAUDIO_EXT_AVAILABLE
wave = torch.tensor([[0.1, -0.2, 0.3, -0.4]])
coefficients = torch.tensor([1., 0.])
filtered = torchaudio.functional.lfilter(wave, coefficients, coefficients)
torch.testing.assert_close(filtered, wave)
boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]])
kept = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8]), 0.5)
assert kept.tolist() == [0]
"""],
            check=True,
            timeout=180,
        )


if __name__ == "__main__":
    unittest.main()
