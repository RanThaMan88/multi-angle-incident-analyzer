"""
Multi-Angle Incident Analyzer
"""
import os
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import numpy as np
from dataclasses import dataclass

import librosa
from scipy.signal import correlate
import ffmpeg
from pyannote.audio import Pipeline
import whisper
import yaml
from loguru import logger

logger.remove()
logger.add("logs/analyzer.log", level="INFO")

@dataclass
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    text: str
    confidence: float = 1.0

class VideoSynchronizer:
    def __init__(self, sr: int = 16000):
        self.sr = sr
        logger.info(f"VideoSynchronizer initialized")
    
    def extract_audio(self, video_path: str, audio_path: str) -> str:
        logger.info(f"Extracting audio from {Path(video_path).name}")
        try:
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream.audio, audio_path, acodec='pcm_s16le', ac=1, ar=self.sr)
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True)
            return audio_path
        except Exception as e:
            logger.error(f"Failed: {e}")
            raise
    
    def find_sync_offset(self, audio1: np.ndarray, audio2: np.ndarray) -> float:
        logger.info("Finding sync offset...")
        audio1 = audio1 / (np.max(np.abs(audio1)) + 1e-8)
        audio2 = audio2 / (np.max(np.abs(audio2)) + 1e-8)
        
        max_samples = int(self.sr * 30)
        correlation = correlate(audio1[:max_samples], audio2[:max_samples], mode='valid')
        lag = np.argmax(np.abs(correlation)) - len(audio1[:max_samples]) + 1
        offset = lag / self.sr
        
        return offset
    
    def sync_videos(self, video_paths: Dict[str, str]) -> Dict:
        logger.info("Starting video synchronization...")
        os.makedirs("./temp", exist_ok=True)
        sync_info = {}
        
        audio_arrays = {}
        for angle, path in video_paths.items():
            audio_path = f"./temp/{angle}_audio.wav"
            self.extract_audio(path, audio_path)
            audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
            audio_arrays[angle] = audio
            sync_info[angle] = {'offset': 0.0, 'audio_path': audio_path, 'synced': True}
        
        primary_angle = list(video_paths.keys())[0]
        primary_audio = audio_arrays[primary_angle]
        
        for angle, audio in audio_arrays.items():
            if angle != primary_angle:
                offset = self.find_sync_offset(primary_audio, audio)
                sync_info[angle]['offset'] = offset
        
        return sync_info

class SpeakerDiarizer:
    def __init__(self):
        logger.info("Loading diarization model...")
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        try:
            self.pipeline.to("cuda")
        except:
            pass
    
    def diarize(self, audio_path: str) -> List[TranscriptSegment]:
        logger.info(f"Diarizing: {Path(audio_path).name}")
        diarization = self.pipeline(audio_path)
        segments = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(TranscriptSegment(
                start=turn.start,
                end=turn.end,
                speaker=speaker,
                text="",
                confidence=1.0
            ))
        
        logger.info(f"Diarization complete: {len(segments)} segments")
        return segments

class AudioTranscriber:
    def __init__(self, model: str = "base"):
        logger.info(f"Loading Whisper model: {model}")
        self.model = whisper.load_model(model)
    
    def transcribe(self, audio_path: str) -> Dict:
        logger.info(f"Transcribing: {Path(audio_path).name}")
        result = self.model.transcribe(audio_path, verbose=False, fp16=True)
        logger.info("Transcription complete")
        return result
    
    def merge_with_diarization(self, transcript: Dict, diarized_segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        logger.info("Merging diarization with transcription...")
        
        speaker_map = {}
        for segment in diarized_segments:
            for t in np.linspace(segment.start, segment.end, 10):
                speaker_map[t] = segment.speaker
        
        merged_segments = []
        for chunk in transcript['segments']:
            start = chunk['start']
            end = chunk['end']
            text = chunk['text'].strip()
            
            mid_time = (start + end) / 2
            speaker = "Unknown"
            
            if speaker_map:
                closest_time = min(speaker_map.keys(), key=lambda t: abs(t - mid_time))
                if abs(closest_time - mid_time) < 5.0:
                    speaker = speaker_map[closest_time]
            
            merged_segments.append(TranscriptSegment(
                start=start,
                end=end,
                speaker=speaker,
                text=text
            ))
        
        return merged_segments

class IncidentAnalyzer:
    def __init__(self):
        self.syncer = VideoSynchronizer()
        self.diarizer = SpeakerDiarizer()
        self.transcriber = AudioTranscriber()
        logger.info("Analyzer initialized")
    
    def analyze_incident(self, video_paths: Dict[str, str], case_name: str = "Incident") -> Dict:
        logger.info(f"Starting analysis: {case_name}")
        
        sync_info = self.syncer.sync_videos(video_paths)
        primary_audio = list(sync_info.values())[0]['audio_path']
        diarized_segments = self.diarizer.diarize(primary_audio)
        transcript_dict = self.transcriber.transcribe(primary_audio)
        merged_transcript = self.transcriber.merge_with_diarization(transcript_dict, diarized_segments)
        
        logger.info("Analysis complete")
        
        return {
            'success': True,
            'transcript': merged_transcript,
            'sync_info': sync_info,
            'case_name': case_name
        }

if __name__ == "__main__":
    analyzer = IncidentAnalyzer()
    print("Ready")