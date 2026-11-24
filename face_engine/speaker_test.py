import os
import wave
import subprocess
import tempfile

import torch
import pyaudio

from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy


MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "pretrained_models",
    "spkrec-ecapa-voxceleb"
)


class SpeakerEngine:

    def __init__(self):

        print(
            "Loading speaker recognition model...",
            flush=True
        )

        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=MODEL_DIR,
            local_strategy=LocalStrategy.COPY
        )

        print(
            "Speaker recognition model loaded.",
            flush=True
        )

    # =====================================================
    # RECORD VOICE
    # =====================================================

    def record_voice(
        self,
        filename,
        duration=5,
        sample_rate=16000
    ):

        print("\nRecording...", flush=True)

        print(
            f"Speak normally for {duration} seconds.",
            flush=True
        )

        audio = pyaudio.PyAudio()

        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=sample_rate,
            input=True,
            frames_per_buffer=1024
        )

        frames = []

        try:

            for _ in range(
                int(sample_rate / 1024 * duration)
            ):

                data = stream.read(
                    1024,
                    exception_on_overflow=False
                )

                frames.append(data)

        finally:

            stream.stop_stream()
            stream.close()

        sample_width = audio.get_sample_size(
            pyaudio.paInt16
        )

        audio.terminate()

        with wave.open(filename, "wb") as wf:

            wf.setnchannels(1)

            wf.setsampwidth(
                sample_width
            )

            wf.setframerate(
                sample_rate
            )

            wf.writeframes(
                b"".join(frames)
            )

        print(
            "Recording saved:",
            filename,
            flush=True
        )

        return filename

    # =====================================================
    # CONVERT WEBM / OTHER AUDIO TO WAV
    # =====================================================

    def convert_to_wav(
        self,
        input_file
    ):

        input_file = os.path.abspath(
            input_file
        )

        if not os.path.exists(input_file):

            raise FileNotFoundError(
                f"Audio file does not exist: {input_file}"
            )

        extension = os.path.splitext(
            input_file
        )[1].lower()

        # -------------------------------------------------
        # Already WAV
        # -------------------------------------------------

        if extension == ".wav":

            return input_file

        # -------------------------------------------------
        # Temporary WAV
        # -------------------------------------------------

        temp_directory = tempfile.mkdtemp(
            prefix="student_voice_"
        )

        output_file = os.path.join(
            temp_directory,
            "converted.wav"
        )

        # -------------------------------------------------
        # FFmpeg conversion
        # -------------------------------------------------

        command = [

            "ffmpeg",

            "-y",

            "-i",
            input_file,

            "-ac",
            "1",

            "-ar",
            "16000",

            "-sample_fmt",
            "s16",

            output_file

        ]

        try:

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

        except FileNotFoundError:

            raise RuntimeError(
                "FFmpeg is not installed or "
                "not available in PATH."
            )

        if result.returncode != 0:

            raise RuntimeError(
                "FFmpeg audio conversion failed:\n"
                + result.stderr
            )

        if not os.path.exists(output_file):

            raise RuntimeError(
                "FFmpeg did not create the WAV file."
            )

        return output_file

    # =====================================================
    # GENERATE SPEAKER EMBEDDING
    # =====================================================

    def get_embedding(self, filename):

        # -------------------------------------------------
        # Convert browser WebM → WAV
        # -------------------------------------------------

        wav_file = self.convert_to_wav(
            filename
        )

        try:

            # -------------------------------------------------
            # Read WAV
            # -------------------------------------------------

            with wave.open(
                wav_file,
                "rb"
            ) as wf:

                channels = wf.getnchannels()

                sample_width = wf.getsampwidth()

                sample_rate = wf.getframerate()

                frame_count = wf.getnframes()

                audio_bytes = wf.readframes(
                    frame_count
                )

            # -------------------------------------------------
            # Validate format
            # -------------------------------------------------

            if sample_width != 2:

                raise ValueError(
                    f"Expected 16-bit audio, "
                    f"got {sample_width * 8}-bit"
                )

            if sample_rate != 16000:

                raise ValueError(
                    f"Expected 16000 Hz audio, "
                    f"got {sample_rate} Hz"
                )

            # -------------------------------------------------
            # PCM → Tensor
            # -------------------------------------------------

            audio = torch.frombuffer(
                audio_bytes,
                dtype=torch.int16
            ).clone()

            audio = audio.float() / 32768.0

            # -------------------------------------------------
            # Stereo → Mono
            # -------------------------------------------------

            if channels > 1:

                audio = audio.reshape(
                    -1,
                    channels
                ).mean(dim=1)

            # -------------------------------------------------
            # [time] → [batch, time]
            # -------------------------------------------------

            waveform = audio.unsqueeze(0)

            # -------------------------------------------------
            # Generate embedding
            # -------------------------------------------------

            with torch.no_grad():

                embedding = self.model.encode_batch(
                    waveform
                )

            return embedding.squeeze()

        finally:

            # -------------------------------------------------
            # Delete converted temporary WAV
            # -------------------------------------------------

            if wav_file != filename:

                try:

                    if os.path.exists(wav_file):

                        os.remove(wav_file)

                    temp_directory = os.path.dirname(
                        wav_file
                    )

                    if os.path.isdir(
                        temp_directory
                    ):

                        os.rmdir(
                            temp_directory
                        )

                except Exception:
                    pass

    # =====================================================
    # COMPARE TWO SPEAKERS
    # =====================================================

    def compare_voices(
        self,
        voice1,
        voice2
    ):

        embedding1 = self.get_embedding(
            voice1
        )

        embedding2 = self.get_embedding(
            voice2
        )

        similarity = torch.nn.functional.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0)
        ).item()

        return similarity


if __name__ == "__main__":

    engine = SpeakerEngine()

    print(
        "\n================================"
    )

    print(
        "Speaker Recognition Test"
    )

    print(
        "================================"
    )

    voice1 = engine.record_voice(
        filename="voice_1.wav",
        duration=5
    )

    print(
        "\nFirst recording completed."
    )

    input(
        "\nPress ENTER when ready "
        "for the second recording..."
    )

    voice2 = engine.record_voice(
        filename="voice_2.wav",
        duration=5
    )

    print(
        "\nSecond recording completed."
    )

    print(
        "\nComparing speakers..."
    )

    score = engine.compare_voices(
        voice1,
        voice2
    )

    print(
        "\n================================"
    )

    print(
        "RESULT"
    )

    print(
        "================================"
    )

    print(
        "Speaker similarity:",
        round(score, 4)
    )

    if score >= 0.70:

        print(
            "Speaker verification: "
            "SAME SPEAKER"
        )

    else:

        print(
            "Speaker verification: "
            "DIFFERENT SPEAKER"
        )