"""
DataLoader for Audio-based Valence-Arousal Extraction with Full Audio
Uses EMOCA image-based predictions as ground truth labels (from CSV files)
Extracts FULL audio files and computes AVERAGE VA across all frames
Searches directory for multiple CSV files (all directions)
One sample per CSV file
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
    Dataset for training audio-based VA extraction with full audio
    Ground truth: EMOCA image-based VA predictions from CSV files (averaged across all frames)
    Input: Full audio file (entire recording)
    Output: Average valence and arousal across all frames in the CSV
    """
    
    def __init__(
        self,
        csv_dir: str,  # Directory containing CSV files with VA annotations
        audio_base_dir: str,  # Base audio directory (e.g., /home/chan9/Continuous-EMOTE/audio)
        sequence_num: str = "001",  # Audio sequence number: "001", "002", etc.
        direction_filter: Optional[str] = None,  # Optional: Only load CSVs with this direction
        audio_sample_rate: int = 16000,
        target_sample_rate: Optional[int] = None,
        mode: str = 'train'
    ):
        """
        Args:
            csv_dir: Directory containing CSV files with EMOCA predictions
            audio_base_dir: Base audio directory (e.g., /path/to/audio)
            sequence_num: Audio sequence number (e.g., "001", "002", "003" for 001.m4a, 002.m4a, etc.)
            direction_filter: Optional - Only load CSVs matching this direction (e.g., "front", "down", etc.). If None, loads all directions.
            audio_sample_rate: Original audio sample rate
            target_sample_rate: Resample audio to this rate (None = keep original)
            mode: 'train', 'val', or 'test'
        """
        self.audio_base_dir = Path(audio_base_dir)
        self.csv_dir = Path(csv_dir)
        self.sequence_num = sequence_num
        self.direction_filter = direction_filter.lower() if direction_filter else None
        self.audio_sample_rate = audio_sample_rate
        self.target_sample_rate = target_sample_rate or audio_sample_rate
        self.mode = mode
        
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
        One sample per CSV file: full audio + average VA
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
            
            # Calculate average VA across all valid rows in CSV
            valid_valences = []
            valid_arousals = []
            expressions = []
            
            for idx, row in df.iterrows():
                if pd.notna(row['valence']) and pd.notna(row['arousal']):
                    valid_valences.append(float(row['valence']))
                    valid_arousals.append(float(row['arousal']))
                    if 'expression' in row and pd.notna(row['expression']):
                        expressions.append(str(row['expression']))
            
            if len(valid_valences) == 0:
                print(f"  WARNING: No valid VA annotations found, skipping...")
                continue
            
            # Compute averages
            avg_valence = float(np.mean(valid_valences))
            avg_arousal = float(np.mean(valid_arousals))
            
            # Get most common expression (if available)
            if expressions:
                from collections import Counter
                most_common_expression = Counter(expressions).most_common(1)[0][0]
            else:
                most_common_expression = 'unknown'
            
            # Create one sample per CSV (full audio + average VA)
            all_samples.append({
                'video_id': video_id,
                'valence': avg_valence,
                'arousal': avg_arousal,
                'expression': most_common_expression,
                'audio_path': audio_path,
                'direction': direction,
                'emotion_category': emotion,
                'level': level,
                'num_frames': len(valid_valences)
            })
            
            print(f"  Average VA: valence={avg_valence:.3f}, arousal={avg_arousal:.3f} (from {len(valid_valences)} frames)")
        
        print(f"\nTotal samples (CSV files): {len(all_samples)}")
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
    
    def _load_full_audio(self, audio_path: Path) -> torch.Tensor:
        """
        Load the full audio file
        Args:
            audio_path: Path to audio file
        Returns:
            Audio tensor of shape (num_samples,) - full audio
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
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform.squeeze(0)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - audio: (num_samples,) full audio waveform
                - valence: scalar average valence value
                - arousal: scalar average arousal value
                - video_id: video identifier
                - expression: most common expression label
                - direction: camera direction
                - emotion_category: emotion category
                - level: intensity level
        """
        sample = self.samples[idx]
        
        # Load full audio
        audio = self._load_full_audio(sample['audio_path'])
        
        return {
            'audio': audio,  # (num_samples,) - variable length
            'valence': torch.tensor(sample['valence'], dtype=torch.float32),
            'arousal': torch.tensor(sample['arousal'], dtype=torch.float32),
            'video_id': sample['video_id'],
            'expression': sample['expression'],
            'direction': sample['direction'],
            'emotion_category': sample['emotion_category'],
            'level': sample['level'],
            'num_frames': sample['num_frames']
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching variable-length audio
    Pads audio to the longest sample in the batch
    """
    # Find max length in batch
    max_length = max(item['audio'].shape[0] for item in batch)
    
    # Pad all audio to max length
    padded_audio = []
    for item in batch:
        audio = item['audio']
        if audio.shape[0] < max_length:
            padding = torch.zeros(max_length - audio.shape[0])
            audio = torch.cat([audio, padding])
        padded_audio.append(audio)
    
    return {
        'audio': torch.stack(padded_audio),  # (batch, max_length)
        'valence': torch.stack([item['valence'] for item in batch]),
        'arousal': torch.stack([item['arousal'] for item in batch]),
        'video_id': [item['video_id'] for item in batch],
        'expression': [item['expression'] for item in batch],
        'direction': [item['direction'] for item in batch],
        'emotion_category': [item['emotion_category'] for item in batch],
        'level': [item['level'] for item in batch],
        'num_frames': torch.tensor([item['num_frames'] for item in batch])
    }


def create_dataloader(
    csv_dir: str,
    audio_base_dir: str,
    sequence_num: str = "001",
    direction_filter: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    Create DataLoader for audio VA training with full audio files
    
    Args:
        csv_dir: Directory containing CSV files with EMOCA predictions
        audio_base_dir: Base audio directory (e.g., /home/chan9/Continuous-EMOTE/audio)
        sequence_num: Audio sequence number (e.g., "001" for 001.m4a, "002" for 002.m4a)
        direction_filter: Optional - Only load CSVs matching this direction (e.g., "front", "down", etc.). If None, loads all directions.
        batch_size: Batch size (smaller recommended due to full audio size)
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        **dataset_kwargs: Additional arguments for AudioVADataset
    
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
    # Full audio with average VA (one sample per CSV file)
    train_loader = create_dataloader(
        csv_dir="/home/chan9/Continuous-EMOTE/emotion_csv_extracted",
        audio_base_dir="/home/chan9/Continuous-EMOTE/audio",
        sequence_num="001",  # Use 001.m4a for all emotions
        direction_filter=None,  # Load all directions
        batch_size=8,  # Smaller batch size due to full audio
        audio_sample_rate=16000,
        mode='train'
    )
    
    # Test loading
    print("\nTesting dataloader...")
    for batch in train_loader:
        print(f"Audio shape: {batch['audio'].shape}")
        print(f"Valence shape: {batch['valence'].shape}")
        print(f"Arousal shape: {batch['arousal'].shape}")
        print(f"Valence range: [{batch['valence'].min():.3f}, {batch['valence'].max():.3f}]")
        print(f"Arousal range: [{batch['arousal'].min():.3f}, {batch['arousal'].max():.3f}]")
        print(f"Video IDs (first 3): {batch['video_id'][:3]}")
        print(f"Expressions (first 3): {batch['expression'][:3]}")
        print(f"Directions (first 3): {batch['direction'][:3]}")
        print(f"Num frames (first 3): {batch['num_frames'][:3].tolist()}")
        break
    
    # Verify individual samples
    print("\n" + "="*60)
    print("VERIFICATION: Sample details")
    print("="*60)
    
    dataset = train_loader.dataset
    print(f"\nDataset has {len(dataset)} samples (CSV files)")
    
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        audio_duration = sample['audio'].shape[0] / 16000  # Assuming 16kHz
        
        print(f"\nSample {i}:")
        print(f"  Video ID: {sample['video_id']}")
        print(f"  Direction: {sample['direction']}")
        print(f"  Emotion: {sample['emotion_category']}, Level: {sample['level']}")
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Audio samples: {sample['audio'].shape[0]}")
        print(f"  Average Valence: {sample['valence']:.3f}")
        print(f"  Average Arousal: {sample['arousal']:.3f}")
        print(f"  Most common expression: {sample['expression']}")
        print(f"  Number of frames used: {sample['num_frames']}")
    
    print("\n" + "="*60)