import os
import torch
import torchaudio
import torchvision
import torch.nn.functional as F
import sentencepiece as spm

from python_speech_features import logfbank
import numpy as np


_SPM_DECODER = None
_UNITS_ID2PIECE = None


def _get_spm_decoder():
    """Lazily initialize SentencePiece decoder if model file exists."""
    global _SPM_DECODER
    if _SPM_DECODER is not None:
        return _SPM_DECODER

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, "spm", "unigram", "unigram5000.model"),
        os.path.join(repo_root, "auto_avsr", "spm", "unigram", "unigram5000.model"),
    ]

    for model_path in candidates:
        if os.path.exists(model_path):
            _SPM_DECODER = spm.SentencePieceProcessor(model_file=model_path)
            return _SPM_DECODER
    return None


def _get_units_id2piece():
    """Load id->piece mapping from *_units.txt used during tokenization."""
    global _UNITS_ID2PIECE
    if _UNITS_ID2PIECE is not None:
        return _UNITS_ID2PIECE

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, "spm", "unigram", "unigram5000_units.txt"),
        os.path.join(repo_root, "auto_avsr", "spm", "unigram", "unigram5000_units.txt"),
    ]

    id2piece = {}
    for units_path in candidates:
        if not os.path.exists(units_path):
            continue
        with open(units_path, "r", encoding="utf8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                piece = parts[0]
                try:
                    idx = int(parts[-1])
                except ValueError:
                    continue
                id2piece[idx] = piece
        if id2piece:
            _UNITS_ID2PIECE = id2piece
            return _UNITS_ID2PIECE

    _UNITS_ID2PIECE = None
    return _UNITS_ID2PIECE


def _maybe_decode_spm_token_ids(text):
    """Decode numeric token-id strings with SentencePiece; return original text otherwise."""
    if not text:
        return text
    parts = text.strip().split()
    if not parts:
        return text

    # Decode only when the entire string is integer token ids.
    if not all(p.lstrip("-").isdigit() for p in parts):
        return text

    try:
        ids = [int(p) for p in parts]

        # 1) Preferred path: decode with units id->piece mapping used by training scripts.
        id2piece = _get_units_id2piece()
        if id2piece is not None:
            pieces = []
            for idx in ids:
                if idx in (-1, 0):  # ignore_id / blank
                    continue
                piece = id2piece.get(idx)
                if piece is None or piece in {"<eos>", "<blank>"}:
                    continue
                pieces.append(piece)

            if pieces:
                decoded = "".join(pieces).replace("▁", " ").replace("<space>", " ").strip().lower()
                if decoded:
                    return decoded

        # 2) Fallback: direct SentencePiece id decoding if available.
        decoder = _get_spm_decoder()
        if decoder is not None:
            decoded = decoder.DecodeIds(ids).strip().lower()
            if decoded:
                return decoded

        return text
    except Exception:
        return text

def stacker(feats, stack_order):
            """
            Concatenating consecutive audio frames
            Args:
            feats - numpy.ndarray of shape [T, F]
            stack_order - int (number of neighboring frames to concatenate
            Returns:
            feats - numpy.ndarray of shape [T', F']
            """
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
            return feats


def cut_or_pad(data, size, dim=0):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(data, (0, 0, 0, padding), "constant")
        size = data.size(dim)
    elif data.size(dim) > size:
        data = data[:size]
    assert data.size(dim) == size
    return data

def load_video(path):
    """
    rtype: torch, T x C x H x W
    """
    vid = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0]
    vid = vid.permute((0, 3, 1, 2))
    return vid


def _resolve_audio_path(video_path):
    """Resolve the corresponding audio path for a video path.

    Supports both:
    - same-folder layout: .../xxx.mp4 -> .../xxx.wav
    - split layout: .../video/.../xxx.mp4 -> .../audio/.../xxx.wav
    """
    # Base candidate: same directory/stem, .wav extension
    base_wav = os.path.splitext(video_path)[0] + ".wav"

    candidates = [base_wav]

    # Common split layouts
    candidates.append(base_wav.replace("/video/", "/audio/"))
    candidates.append(base_wav.replace("_video_seg16s", "_audio_seg16s"))
    candidates.append(base_wav.replace("/video_seg16s/", "/audio_seg16s/"))

    # De-duplicate while preserving order
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            unique_candidates.append(c)
            seen.add(c)

    for candidate in unique_candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find matching audio file for video path: {video_path}. "
        f"Checked: {unique_candidates}"
    )


def load_audio(path):
    """
    rtype: torch, T x 1
    """
    audio_path = _resolve_audio_path(path)
    waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
    return waveform.transpose(1, 0)


def _resolve_text_path(video_path):
    """Resolve transcript .txt path corresponding to a video path."""
    base_txt = os.path.splitext(video_path)[0] + ".txt"
    candidates = [
        base_txt,
        base_txt.replace("/lrs3/video/", "/lrs3/"),
        base_txt.replace("/lrs2/video/", "/lrs2/"),
        base_txt.replace("/video/", "/text/"),
        base_txt.replace("_video_seg16s", "_text_seg16s"),
        base_txt.replace("/video_seg16s/", "/text_seg16s/"),
        base_txt.replace("lrs3_video_seg16s", "lrs3_text_seg16s"),
        base_txt.replace("lrs2_video_seg16s", "lrs2_text_seg16s"),
    ]

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    return None


def _extract_transcript_text(raw_text):
    """Extract clean utterance text from transcript file content.

    Handles common formats such as:
      text: hello world
      conf: 6
    """
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if not lines:
        return ""

    # Prefer explicit `text:` line when present
    for ln in lines:
        if ln.lower().startswith("text:"):
            return ln.split(":", 1)[1].strip().lower()

    # Otherwise use first non-metadata line
    for ln in lines:
        low = ln.lower()
        if not (low.startswith("conf:") or low.startswith("confidence:")):
            return ln.lower()

    # Fallback: first line
    return lines[0].lower()



# Note: since the number of tokens produced by audio encoders like WavLM and Whisper can vary a little, we can't truncate the
# audio samples here as we do for video such that we get an exact number of tokens compatible with the downsample ratios.

class AVDataset_LLM(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        label_path,
        subset,
        modality,
        audio_transform,
        video_transform,
        rate_ratio=640,
        downsample_ratio = None,
        is_avhubert_audio = False,
        single_projector_avhubert = None
    ):

        self.root_dir = root_dir

        self.modality = modality
        self.rate_ratio = rate_ratio
        
        self.audio_transform = audio_transform
        self.video_transform = video_transform
        
        self.is_avhubert_audio = is_avhubert_audio
        self.single_projector_avhubert = single_projector_avhubert
        
        self.list = self.load_list(label_path)
        self.input_lengths = [int(_[2]) for _ in self.list]
       
        if modality == "video" or modality == "audiovisual" or modality == "audiovisual_avhubert":
            self.downsample_video = downsample_ratio if downsample_ratio != 1 else None 
        
    def load_list(self, label_path):
        paths_counts_labels = []
        for path_count_label in open(label_path).read().splitlines():
            parts = path_count_label.split(",")
            if len(parts) == 5:
                # LRS3 format: dataset_name, rel_path, input_length, _, text
                dataset_name, rel_path, input_length, _, text = parts
            elif len(parts) == 4:
                # LRS2 format: dataset_name, rel_path, input_length, text
                dataset_name, rel_path, input_length, text = parts
            else:
                raise ValueError(f"Unexpected CSV format. Expected 4 or 5 columns, got {len(parts)}: {path_count_label}")
            paths_counts_labels.append((dataset_name, rel_path, input_length, text))
        return paths_counts_labels

    def __getitem__(self, idx):
        dataset_name, rel_path, _, text = self.list[idx]
        path = os.path.join(self.root_dir, dataset_name, rel_path)
        
        # Prefer raw transcript text file when present; otherwise use CSV text.
        text_path = _resolve_text_path(path)

        if text_path and os.path.exists(text_path):
            with open(text_path, 'r') as f:
                text = _extract_transcript_text(f.read())
        else:
            text = _maybe_decode_spm_token_ids(text)
        # If text file doesn't exist, fall back to text from CSV (which might be tokenized)
        
        if self.modality == "video":
            video = load_video(path)
            video = self.video_transform(video)
            
            if self.downsample_video:
                video = video[: video.size(0) // self.downsample_video * self.downsample_video]
            
            return {"video": video, "tokens": text}
        elif self.modality == "audio":
            audio = load_audio(path)
            
            audio = self.audio_transform(audio)
            
            if self.is_avhubert_audio:
                device = audio.device
                audio = logfbank(audio)
                audio = torch.tensor(stacker(audio, 4), dtype= torch.float32, device= device)
                with torch.no_grad():
                    audio = F.layer_norm(audio, audio.shape[1:])
                  
            return {"audio": audio, "tokens": text}
        elif self.modality == "audiovisual":
            video = load_video(path)
            audio = load_audio(path)
            audio = cut_or_pad(audio, len(video) * self.rate_ratio)
            
            video = self.video_transform(video)
            audio = self.audio_transform(audio)
            
            if self.downsample_video:
                video = video[: video.size(0) // self.downsample_video * self.downsample_video]
                
            return {"video": video, "audio": audio, "tokens": text}
        elif self.modality == "audiovisual_avhubert":
            assert self.is_avhubert_audio == True
            
            video = load_video(path)
            audio = load_audio(path)
            audio = cut_or_pad(audio, len(video) * self.rate_ratio)
            
            video = self.video_transform(video)
            
            if not self.single_projector_avhubert:
                video = video[: video.size(0) // self.downsample_video * self.downsample_video]
             
            device = audio.device
            audio = logfbank(audio)
            audio = torch.tensor(stacker(audio, 4), dtype= torch.float32, device= device)
            with torch.no_grad():
                audio = F.layer_norm(audio, audio.shape[1:])
            return {"video": video, "audio": audio, "tokens": text}

    def __len__(self):
        return len(self.list)
