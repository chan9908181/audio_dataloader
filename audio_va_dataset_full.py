"""
DataLoader for Audio-based Valence-Arousal Extraction with Sliding Window
Uses EMOCA image-based predictions as ground truth labels (from CSV files)
Extracts 1-second audio windows (0.5s before + 0.5s after) for EVERY frame
Searches directory for multiple CSV files (all directions)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import re
import subprocess
import tempfile
import os

# Try to set backend for m4a support (works in newer torchaudio versions)
try:
    torchaudio.set_audio_backend("ffmpeg")
except (AttributeError, RuntimeError):
    # Fallback for older versions - will use default backend
    pass


class AudioVADataset(Dataset):
    """
    Dataset for training audio-based VA extraction with sliding window
    Ground truth: EMOCA image-based VA predictions from CSV files (interpolated for all frames)
    Input: 1-second audio windows centered on each frame
    """
    
    def __init__(
        self,
        csv_dir: str,  # Directory containing CSV files with VA annotations
        audio_base_dir: str,  # Base audio directory (e.g., /home/chan9/Continuous-EMOTE/audio)
        sequence_num: str = "001",  # Audio sequence number: "001", "002", etc.
        direction_filter: Optional[str] = None,  # Optional: Only load CSVs with this direction
        fps: int = 30,
        audio_sample_rate: int = 16000,
        window_duration: float = 1.0,  # Total window in seconds
        target_sample_rate: Optional[int] = None,
        mode: str = 'train',
        interpolate_va: bool = True,  # Interpolate VA for frames between annotations
        use_all_frames: bool = True  # Generate samples for all frames, not just annotated ones
    ):
        """
        Args:
            csv_dir: Directory containing CSV files with EMOCA predictions
            audio_base_dir: Base audio directory (e.g., /path/to/audio)
            sequence_num: Audio sequence number (e.g., "001", "002", "003" for 001.m4a, 002.m4a, etc.)
            direction_filter: Optional - Only load CSVs matching this direction (e.g., "front", "down", etc.). If None, loads all directions.
            fps: Video frames per second
            audio_sample_rate: Original audio sample rate
            window_duration: Total audio window duration (default 1.0s)
            target_sample_rate: Resample audio to this rate (None = keep original)
            mode: 'train', 'val', or 'test'
            interpolate_va: Whether to interpolate VA values for frames between annotations
            use_all_frames: If True, generate samples for ALL frames using sliding window
        """
        self.audio_base_dir = Path(audio_base_dir)
        self.csv_dir = Path(csv_dir)
        self.sequence_num = sequence_num
        self.direction_filter = direction_filter.lower() if direction_filter else None
        self.fps = fps
        self.audio_sample_rate = audio_sample_rate
        self.target_sample_rate = target_sample_rate or audio_sample_rate
        self.window_duration = window_duration
        self.half_window_duration = window_duration / 2.0
        self.mode = mode
        self.interpolate_va = interpolate_va
        self.use_all_frames = use_all_frames
        
        # Calculate samples per window
        self.samples_per_window = int(self.target_sample_rate * self.window_duration)
        self.half_window_samples = self.samples_per_window // 2
        
        # Find all CSV files in directory
        self.csv_files = self._find_csv_files()
        if self.direction_filter:
            print(f"Found {len(self.csv_files)} CSV files matching direction '{direction_filter}'")
        else:
            print(f"Found {len(self.csv_files)} CSV files (all directions)")
        
        # Load all CSVs and build dataset index
        self.samples = self._build_dataset_index()
        
        print(f"Built dataset with {len(self.samples)} samples")
        
    def _find_csv_files(self) -> List[Path]:
        """
        Find all CSV files in the directory
        CSV format: {direction}_{emotion}_{level}_emotion_data.csv
        Example: front_angry_level_1_emotion_data.csv
        """
        csv_files = []
        
        # Search for all CSV files in directory
        for csv_file in self.csv_dir.glob('*_emotion_data.csv'):
            # Parse direction from filename
            filename = csv_file.stem  # Remove .csv extension
            parts = filename.replace('_emotion_data', '').split('_')
            
            if len(parts) >= 1:
                direction = parts[0].lower()
                
                # Filter by direction if specified
                if self.direction_filter is None or direction == self.direction_filter:
                    csv_files.append(csv_file)
                    print(f"  Found: {csv_file.name}")
        
        return sorted(csv_files)  # Sort for consistent ordering
        
    def _parse_frame_number(self, image_name) -> Optional[int]:
        """
        Parse frame number from image name
        Format: 000100, 000200, 000300 (where each increment of 100 = 1 frame)
        Actual frame index = value // 100
        Note: pandas may read this as int64 or str
        """
        try:
            # Handle both int and str types
            frame_value = int(image_name)
            # Convert 3100 -> frame 31, 100 -> frame 1, etc.
            return frame_value // 100
        except (ValueError, TypeError):
            return None
    
    def _parse_video_id_from_csv(self, csv_path: Path) -> Tuple[str, str, str]:
        """
        Extract video info from CSV filename
        Handles multi-part directions like left_30, right_60, etc.
        CSV format: front_angry_level_1_emotion_data.csv, left_30_angry_level_2_emotion_data.csv, etc.

        Returns: (direction, emotion, level)
        """
        csv_filename = csv_path.stem  # Remove .csv extension
        parts = csv_filename.replace('_emotion_data', '').split('_')

        # Handle multi-part directions like left_30, right_60, etc.
        if len(parts) >= 4 and parts[1].isdigit():
            direction = f"{parts[0]}_{parts[1]}"  # e.g., left_30
            emotion = parts[2]
            level = '_'.join(parts[3:])  # e.g., level_2
        elif len(parts) >= 3:
            direction = parts[0]
            emotion = parts[1]
            level = '_'.join(parts[2:])
        else:
            direction = "unknown"
            emotion = "unknown"
            level = "level_1"

        return direction, emotion, level
    
    def _build_dataset_index(self) -> List[Dict]:
        """
        Build index of all samples from all CSV files
        If use_all_frames=True: Generate samples for EVERY frame using sliding window
        Otherwise: One sample per CSV row
        """
        all_samples = []
        
        for csv_file in self.csv_files:
            print(f"\nProcessing CSV: {csv_file.name}")
            
            # Load CSV
            df = pd.read_csv(csv_file)
            df = df.sort_values('image_name').reset_index(drop=True)
            print(f"  {len(df)} rows in CSV")
            
            # Parse video info from CSV filename
            direction, emotion, level = self._parse_video_id_from_csv(csv_file)
            video_id = f"{direction}_{emotion}_{level}_{self.sequence_num}"
            
            # Find audio file
            audio_path = self._find_audio_file(emotion, level, self.sequence_num)
            
            if audio_path is None:
                print(f"  WARNING: Audio file {self.sequence_num}.m4a not found for {emotion}/{level}, skipping...")
                continue
            
            print(f"  Using audio: {audio_path.name}")
            
            # Get audio duration
            try:
                try:
                    audio_info = torchaudio.info(str(audio_path))
                    audio_duration = audio_info.num_frames / audio_info.sample_rate
                    total_frames = int(audio_duration * self.fps)
                except RuntimeError:
                    # Fallback: Use ffprobe to get duration
                    result = subprocess.run(
                        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                         '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)],
                        capture_output=True, text=True
                    )
                    audio_duration = float(result.stdout.strip())
                    total_frames = int(audio_duration * self.fps)
            except Exception as e:
                print(f"  WARNING: Error loading audio: {e}, skipping...")
                continue
            
            # Build frame annotations dictionary
            frames_dict = {}
            for idx, row in df.iterrows():
                frame_idx = self._parse_frame_number(row['image_name'])
                if frame_idx is None or pd.isna(row['valence']) or pd.isna(row['arousal']):
                    continue
                
                frames_dict[frame_idx] = {
                    'valence': float(row['valence']),
                    'arousal': float(row['arousal']),
                    'expression': row.get('expression', 'unknown'),
                    'image_name': str(row['image_name'])
                }
            
            if len(frames_dict) == 0:
                print(f"  WARNING: No valid frames found, skipping...")
                continue
            
            # Generate samples
            if self.use_all_frames:
                # SLIDING WINDOW: Generate samples for ALL frames
                min_frame = min(frames_dict.keys())
                max_frame = max(frames_dict.keys())
                
                # Interpolate VA values for all frames
                frame_indices = sorted(frames_dict.keys())
                valences = [frames_dict[f]['valence'] for f in frame_indices]
                arousals = [frames_dict[f]['arousal'] for f in frame_indices]
                
                for frame_idx in range(min_frame, min(max_frame + 1, total_frames)):
                    # Calculate frame time and audio window bounds
                    frame_time = frame_idx / self.fps
                    window_start_time = frame_time - self.half_window_duration
                    window_end_time = frame_time + self.half_window_duration
                    
                    # Skip frames where the window would require padding
                    if window_start_time < 0 or window_end_time > audio_duration:
                        continue
                    
                    # Interpolate VA for this frame
                    if self.interpolate_va and frame_idx not in frames_dict:
                        valence = np.interp(frame_idx, frame_indices, valences)
                        arousal = np.interp(frame_idx, frame_indices, arousals)
                        expression = 'interpolated'
                        image_name = f"{frame_idx * 100:08d}"
                    else:
                        # Use exact annotation
                        if frame_idx in frames_dict:
                            valence = frames_dict[frame_idx]['valence']
                            arousal = frames_dict[frame_idx]['arousal']
                            expression = frames_dict[frame_idx]['expression']
                            image_name = frames_dict[frame_idx]['image_name']
                        else:
                            continue
                    
                    all_samples.append({
                        'video_id': video_id,
                        'frame_idx': frame_idx,
                        'valence': valence,
                        'arousal': arousal,
                        'expression': expression,
                        'audio_path': audio_path,
                        'total_frames': total_frames,
                        'image_name': image_name,
                        'direction': direction,
                        'emotion_category': emotion,
                        'level': level
                    })
            else:
                # NO SLIDING WINDOW: Only annotated frames
                for frame_idx, frame_data in frames_dict.items():
                    all_samples.append({
                        'video_id': video_id,
                        'frame_idx': frame_idx,
                        'valence': frame_data['valence'],
                        'arousal': frame_data['arousal'],
                        'expression': frame_data['expression'],
                        'audio_path': audio_path,
                        'total_frames': total_frames,
                        'image_name': frame_data['image_name'],
                        'direction': direction,
                        'emotion_category': emotion,
                        'level': level
                    })
            
            print(f"  Generated {len(all_samples) - len([s for s in all_samples if s['video_id'] != video_id])} samples for this video")
        
        print(f"\nTotal samples across all CSVs: {len(all_samples)}")
        return all_samples
    
    def _find_audio_file(self, emotion: str, level: str, sequence_num: str) -> Optional[Path]:
        """
        Find audio file in structure: audio_base_dir/emotion/level/sequence.m4a
        Example: audio/angry/level_1/001.m4a
        """
        # Try different extensions
        for ext in ['.m4a', '.wav', '.mp3', '.flac']:
            # Primary path: audio/angry/level_1/001.m4a
            audio_path = self.audio_base_dir / emotion / level / f"{sequence_num}{ext}"
            if audio_path.exists():
                return audio_path
            
            # Try without level prefix: audio/angry/1/001.m4a
            level_num = level.replace('level_', '')
            audio_path = self.audio_base_dir / emotion / level_num / f"{sequence_num}{ext}"
            if audio_path.exists():
                return audio_path
        
        return None
    
    def _extract_audio_window(
        self, 
        audio_path: Path, 
        frame_idx: int, 
        total_frames: int
    ) -> torch.Tensor:
        """
        Extract 1-second audio window centered on frame_idx
        Args:
            audio_path: Path to audio file
            frame_idx: Target frame index
            total_frames: Total number of frames in video
        Returns:
            Audio tensor of shape (samples_per_window,)
        """
        # Load full audio - use ffmpeg to convert m4a to wav first
        try:
            # Try with default backend first
            waveform, sr = torchaudio.load(str(audio_path))
        except RuntimeError:
            # torchaudio can't load m4a - use ffmpeg to convert to temp wav
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                # Convert m4a to wav using ffmpeg
                subprocess.run(
                    ['ffmpeg', '-i', str(audio_path), '-ar', str(self.target_sample_rate),
                     '-ac', '1', '-y', tmp_path],
                    capture_output=True, check=True
                )
                # Load the converted wav file
                waveform, sr = torchaudio.load(tmp_path)
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        # Resample if needed
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            waveform = resampler(waveform)
            sr = self.target_sample_rate
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Calculate center sample position
        frame_time = frame_idx / self.fps
        center_sample = int(frame_time * sr)
        
        # Calculate window bounds
        start_sample = center_sample - self.half_window_samples
        end_sample = center_sample + self.half_window_samples
        
        # Extract audio window (no padding needed since we filter frames during dataset building)
        audio_window = waveform[:, start_sample:end_sample]
        
        # Ensure exact length (handle rounding errors)
        if audio_window.shape[1] != self.samples_per_window:
            if audio_window.shape[1] < self.samples_per_window:
                padding = torch.zeros((1, self.samples_per_window - audio_window.shape[1]))
                audio_window = torch.cat([audio_window, padding], dim=1)
            else:
                audio_window = audio_window[:, :self.samples_per_window]
        
        return audio_window.squeeze(0)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - audio: (samples_per_window,) audio window
                - valence: scalar valence value
                - arousal: scalar arousal value
                - frame_idx: frame index
                - video_id: video identifier
                - expression: expression label
        """
        sample = self.samples[idx]
        
        # Extract audio window
        audio_window = self._extract_audio_window(
            sample['audio_path'],
            sample['frame_idx'],
            sample['total_frames']
        )
        
        return {
            'audio': audio_window,  # (samples_per_window,)
            'valence': torch.tensor(sample['valence'], dtype=torch.float32),
            'arousal': torch.tensor(sample['arousal'], dtype=torch.float32),
            'frame_idx': sample['frame_idx'],
            'video_id': sample['video_id'],
            'expression': sample['expression'],
            'image_name': sample['image_name']
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching"""
    return {
        'audio': torch.stack([item['audio'] for item in batch]),
        'valence': torch.stack([item['valence'] for item in batch]),
        'arousal': torch.stack([item['arousal'] for item in batch]),
        'frame_idx': torch.tensor([item['frame_idx'] for item in batch]),
        'video_id': [item['video_id'] for item in batch],  # Keep as list
        'expression': [item['expression'] for item in batch],  # Keep as list
        'image_name': [item['image_name'] for item in batch]  # Keep as list
    }


def create_dataloader(
    csv_dir: str,
    audio_base_dir: str,
    sequence_num: str = "001",
    direction_filter: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    Create DataLoader for audio VA training with sliding window
    
    Args:
        csv_dir: Directory containing CSV files with EMOCA predictions
        audio_base_dir: Base audio directory (e.g., /home/chan9/Continuous-EMOTE/audio)
        sequence_num: Audio sequence number (e.g., "001" for 001.m4a, "002" for 002.m4a)
        direction_filter: Optional - Only load CSVs matching this direction (e.g., "front", "down", etc.). If None, loads all directions.
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        **dataset_kwargs: Additional arguments for AudioVADataset
            - use_all_frames: If True, use sliding window for all frames
            - interpolate_va: If True, interpolate VA between annotations
    
    Returns:
        DataLoader instance
    """
    dataset = AudioVADataset(
        csv_dir=csv_dir,
        audio_base_dir=audio_base_dir,
        sequence_num=sequence_num,
        direction_filter=direction_filter,
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


# Example usage
if __name__ == "__main__":
    # With sliding window (generates samples for ALL frames from ALL CSVs)
    train_loader = create_dataloader(
        csv_dir="/home/chan9/Continuous-EMOTE/emotion_csv_extracted",
        audio_base_dir="/home/chan9/Continuous-EMOTE/audio",
        sequence_num="001",  # Specify which audio file: "001" for 001.m4a, "002" for 002.m4a, etc.
        direction_filter=None,  # Load all directions (front, down, left, right, etc.)
        batch_size=16,
        fps=30,
        audio_sample_rate=16000,
        window_duration=1.0,
        mode='train',
        use_all_frames=True,  # SLIDING WINDOW: sample every frame
        interpolate_va=True   # Interpolate VA for frames between annotations
    )
    
    # Test loading
    print("\nTesting dataloader...")
    for batch in train_loader:
        print(f"Audio shape: {batch['audio'].shape}")
        print(f"Valence shape: {batch['valence'].shape}")
        print(f"Arousal shape: {batch['arousal'].shape}")
        print(f"Valence range: [{batch['valence'].min():.3f}, {batch['valence'].max():.3f}]")
        print(f"Arousal range: [{batch['arousal'].min():.3f}, {batch['arousal'].max():.3f}]")
        print(f"Frame indices (first 10): {batch['frame_idx'][:10].tolist()}")
        print(f"Video IDs (first 3): {batch['video_id'][:3]}")
        print(f"Expressions (first 5): {batch['expression'][:5]}")
        break
    
    # Verify audio window extraction for specific frames
    print("\n" + "="*60)
    print("VERIFICATION: Checking audio window extraction")
    print("="*60)
    
    dataset = train_loader.dataset
    fps = dataset.fps
    window_duration = dataset.window_duration
    
    # Check first few samples with shuffle=False
    print(f"\nDataset has {len(dataset)} samples")
    print(f"FPS: {fps}, Window duration: {window_duration}s")
    
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        frame_idx = sample['frame_idx']  # Already an int
        frame_time = frame_idx / fps
        window_start = frame_time - window_duration/2
        window_end = frame_time + window_duration/2
        
        print(f"\nSample {i}:")
        print(f"  Frame index: {frame_idx}")
        print(f"  Frame time: {frame_time:.3f}s")
        print(f"  Audio window: [{window_start:.3f}s, {window_end:.3f}s]")
        print(f"  Audio shape: {sample['audio'].shape}")
        print(f"  Expected samples: {int(window_duration * dataset.target_sample_rate)}")
        print(f"  Valence: {sample['valence'].item():.3f}, Arousal: {sample['arousal'].item():.3f}")
        
        # Verify window is within bounds (no padding needed)
        if window_start < 0:
            print(f"  ⚠️  WARNING: Window starts before audio (needs padding)")
        else:
            print(f"  ✓ Window start is valid (>= 0s)")
    
    print("\n" + "="*60)