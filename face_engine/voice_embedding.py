import os
import sys
import json
import wave
import shutil
import tempfile
import subprocess
from contextlib import redirect_stdout

import torch
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy


FFMPEG_PATH = (
    r"C:\Users\HP\anaconda3\envs\student"
    r"\Library\bin\ffmpeg.exe"
)

# Add the directory containing FFmpeg to PATH
os.environ["PATH"] = (
    os.path.dirname(FFMPEG_PATH)
    + os.pathsep
    + os.environ.get("PATH", "")
)


def send_json(data):

    print(
        json.dumps(data),
        flush=True
    )


def convert_to_wav(input_file):

    """
    Convert browser audio (WebM/Opus) to:

    WAV
    Mono
    16 kHz
    16-bit PCM
    """

    ffmpeg = FFMPEG_PATH

    if not os.path.isfile(ffmpeg):

        raise RuntimeError(
            f"FFmpeg not found at: {ffmpeg}"
        )

    temp_dir = tempfile.mkdtemp(
        prefix="voice_convert_"
    )

    wav_file = os.path.join(
        temp_dir,
        "voice.wav"
    )

    command = [

        ffmpeg,

        "-y",

        "-i",
        input_file,

        "-ac",
        "1",

        "-ar",
        "16000",

        "-sample_fmt",
        "s16",

        wav_file

    ]

    result = subprocess.run(

        command,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        text=True

    )

    if result.returncode != 0:

        raise RuntimeError(

            "Audio conversion failed: "
            + result.stderr[-1000:]

        )

    return wav_file, temp_dir



def load_wav(filename):

    """
    Load 16-bit PCM WAV.

    Returns:

        [1, time] float32 tensor
    """

    with wave.open(
        filename,
        "rb"
    ) as wf:

        channels = (
            wf.getnchannels()
        )

        sample_width = (
            wf.getsampwidth()
        )

        sample_rate = (
            wf.getframerate()
        )

        frame_count = (
            wf.getnframes()
        )

        audio_bytes = (
            wf.readframes(
                frame_count
            )
        )

    if sample_width != 2:

        raise ValueError(

            f"Expected 16-bit audio, "
            f"got {sample_width * 8}-bit audio."

        )
    if sample_rate != 16000:

        raise ValueError(

            f"Expected 16000 Hz audio, "
            f"got {sample_rate} Hz."

        )
    audio = torch.frombuffer(

        audio_bytes,

        dtype=torch.int16

    ).clone()

    
    audio = (
        audio.float()
        / 32768.0
    )

    if channels > 1:

        audio = audio.reshape(

            -1,

            channels

        ).mean(
            dim=1
        )

    waveform = audio.unsqueeze(0)

    return waveform


def generate_embedding(audio_file):

    model_dir = os.path.join(

        os.path.dirname(__file__),

        "pretrained_models",

        "spkrec-ecapa-voxceleb"

    )
    with open(
        os.devnull,
        "w"
    ) as devnull:

        with redirect_stdout(
            devnull
        ):

            speaker_model = (
                SpeakerRecognition.from_hparams(

                    source=
                    "speechbrain/"
                    "spkrec-ecapa-voxceleb",

                    savedir=
                    model_dir,

                    local_strategy=
                    LocalStrategy.COPY

                )
            )

    waveform = load_wav(
        audio_file
    )

    with torch.no_grad():

        embedding = (
            speaker_model.encode_batch(
                waveform
            )
        )

    embedding = (

        embedding

        .squeeze()

        .detach()

        .cpu()

        .tolist()

    )

    if not embedding:

        raise ValueError(
            "Speaker embedding is empty."
        )

    return embedding


def main():
    if len(sys.argv) != 2:

        send_json({

            "success": False,

            "error":
                "INVALID_ARGUMENTS",

            "message":
                "Voice audio file path is required."

        })

        return 1

    input_file = os.path.abspath(
        sys.argv[1]
    )

    if not os.path.isfile(
        input_file
    ):

        send_json({

            "success": False,

            "error":
                "AUDIO_FILE_NOT_FOUND",

            "message":
                "Voice audio file was not found.",

            "file":
                input_file

        })

        return 1

    wav_file = None
    temp_dir = None

    try:
        wav_file, temp_dir = (
            convert_to_wav(
                input_file
            )
        )
        embedding = (
            generate_embedding(
                wav_file
            )
        )

        send_json({

            "success":
                True,

            "message":
                "Voice embedding generated successfully.",

            "embedding":
                embedding,

            "dimensions":
                len(embedding),

            "sample_rate":
                16000,

            "channels":
                1

        })

        return 0

    except Exception as e:
        send_json({

            "success":
                False,

            "error":
                "VOICE_EMBEDDING_ERROR",

            "message":
                str(e)

        })

        return 1

    finally:
        if (

            temp_dir

            and

            os.path.isdir(
                temp_dir
            )

        ):

            shutil.rmtree(

                temp_dir,

                ignore_errors=True

            )


if __name__ == "__main__":

    sys.exit(
        main()
    )