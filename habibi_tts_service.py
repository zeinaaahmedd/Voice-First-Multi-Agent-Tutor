import inspect
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from cached_path import cached_path
from hydra.utils import get_class
from importlib.resources import files
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import load_model, load_vocoder
from habibitts.infer.utils_infer import preprocess_ref_audio_text, infer_process

logger = logging.getLogger(__name__)


@dataclass
class HabibiTTSConfig:
    model_type: str = "Specialized"
    dialect: str = "EGY"
    ref_audio_path: str = ""
    ref_text: str = ""
    output_dir: str = "TTS/output"
    device: str | None = None
    nfe_step: int | None = None
    cfg_strength: float | None = None
    speed: float | None = None
    target_rms: float | None = None
    use_fp16: bool = False
    remove_silence: bool = False
    max_chars: int = 200


class HabibiTTSService:
    def __init__(self, config: HabibiTTSConfig, load: bool = True):
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_cfg = OmegaConf.load(
            str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml"))
        )
        self.target_sample_rate = self.model_cfg.model.mel_spec.target_sample_rate
        self.ema_model = None
        self.vocoder = None
        self.ref_audio = None
        self.ref_text = None

        if load:
            self._load_model_and_vocoder()
            self._load_reference()

    def _checkpoint_path(self) -> str:
        base = f"hf://SWivid/Habibi-TTS/{self.config.model_type}/{self.config.dialect}"
        return f"{base}/model_100000.safetensors"

    def _vocab_path(self) -> str:
        base = f"hf://SWivid/Habibi-TTS/{self.config.model_type}/{self.config.dialect}"
        return f"{base}/vocab.txt"

    def _load_model_and_vocoder(self) -> None:
        logger.info("Loading Habibi model (%s/%s)...", self.config.model_type, self.config.dialect)
        ckpt_file = cached_path(self._checkpoint_path())
        vocab_file = cached_path(self._vocab_path())

        model_cls = get_class(f"f5_tts.model.{self.model_cfg.model.backbone}")
        model_arc = self.model_cfg.model.arch
        mel_spec_type = self.model_cfg.model.mel_spec.mel_spec_type

        self.ema_model = load_model(
            model_cls,
            model_arc,
            ckpt_file,
            mel_spec_type=mel_spec_type,
            vocab_file=vocab_file,
            device=self.device,
            use_fp16=self.config.use_fp16,
        )
        self.vocoder = load_vocoder(
            vocoder_name=mel_spec_type,
            is_local=False,
            local_path="",
            device=self.device,
        )

    def _load_reference(self) -> None:
        if not self.config.ref_audio_path or not self.config.ref_text:
            raise ValueError("ref_audio_path and ref_text are required")

        kwargs = {
            "target_sample_rate": self.target_sample_rate,
            "device": self.device,
        }
        kwargs = self._filter_kwargs(preprocess_ref_audio_text, kwargs)
        self.ref_audio, self.ref_text = preprocess_ref_audio_text(
            self.config.ref_audio_path,
            self.config.ref_text,
            **kwargs,
        )

    def _filter_kwargs(self, func, kwargs: dict) -> dict:
        params = inspect.signature(func).parameters
        return {k: v for k, v in kwargs.items() if k in params and v is not None}

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        current = []

        for word in words:
            current.append(word)
            if len(" ".join(current)) >= self.config.max_chars:
                chunks.append(" ".join(current).strip())
                current = []

        if current:
            chunks.append(" ".join(current).strip())

        return [c for c in chunks if c]

    def _infer_chunk(self, text: str):
        kwargs = {
            "cfg_strength": self.config.cfg_strength,
            "nfe_step": self.config.nfe_step,
            "speed": self.config.speed,
            "target_rms": self.config.target_rms,
            "remove_silence": self.config.remove_silence,
        }
        kwargs = self._filter_kwargs(infer_process, kwargs)
        return infer_process(
            self.ema_model,
            self.vocoder,
            self.ref_audio,
            self.ref_text,
            text,
            **kwargs,
        )

    def _extract_wave(self, result):
        if isinstance(result, (list, tuple)) and result:
            wave = result[0]
            sample_rate = result[1] if len(result) > 1 else None
            return wave, sample_rate
        if isinstance(result, dict):
            wave = result.get("waveform") or result.get("audio") or result.get("wav")
            sample_rate = result.get("sample_rate")
            return wave, sample_rate
        return result, None

    def synthesize(self, text: str, output_path: str | None = None) -> dict:
        if not text or not text.strip():
            return {
                "success": False,
                "output_path": None,
                "sample_rate": None,
                "text": text,
                "chunks_count": 0,
                "error": "Empty input",
            }

        chunks = self._chunk_text(text)
        waves = []
        sample_rate = None

        for chunk in chunks:
            result = self._infer_chunk(chunk)
            wave, sr = self._extract_wave(result)
            if wave is None:
                raise RuntimeError("Inference returned no audio")
            if sample_rate is None:
                sample_rate = sr or self.target_sample_rate
            if isinstance(wave, torch.Tensor):
                wave_tensor = wave
            else:
                wave_tensor = torch.tensor(wave)
            wave_tensor = wave_tensor.squeeze()
            waves.append(wave_tensor)

        full_wave = torch.cat(waves, dim=-1).unsqueeze(0)
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if output_path is None:
            output_path = str(out_dir / f"habibi_tts_{int(time.time())}.wav")

        torchaudio.save(output_path, full_wave.cpu(), sample_rate)

        return {
            "success": True,
            "output_path": output_path,
            "sample_rate": sample_rate,
            "text": text,
            "chunks_count": len(chunks),
        }
