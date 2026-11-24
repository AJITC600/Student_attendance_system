import os
import cv2
import wave
import torch
import pyaudio

from insightface.app import FaceAnalysis
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy


class RecognitionEngine:

    def __init__(self):

        print("Initializing Recognition Engine...")

        # =================================================
        # FACE RECOGNITION
        # =================================================

        print("\nLoading face recognition...")

        self.face_app = FaceAnalysis(
            name="buffalo_l"
        )

        self.face_app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        print("Face recognition loaded.")

        # =================================================
        # SPEAKER RECOGNITION
        # =================================================

        print("\nLoading speaker recognition...")

        model_dir = os.path.join(
            os.path.dirname(__file__),
            "pretrained_models",
            "spkrec-ecapa-voxceleb"
        )

        self.speaker_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=model_dir,
            local_strategy=LocalStrategy.COPY
        )

        print("Speaker recognition loaded.")

        print("\nRecognition Engine ready.")

    # =====================================================
    # FACE RECOGNITION
    # =====================================================

    def detect_faces(self, frame):
        """
        Detect faces from an OpenCV frame.

        Returns:
            List of InsightFace face objects.
        """

        return self.face_app.get(frame)

    # -----------------------------------------------------
    # Get face embedding
    # -----------------------------------------------------

    def get_face_embedding(self, frame):

        faces = self.detect_faces(frame)

        if len(faces) == 0:
            return None

        # Use the largest detected face
        face = max(
            faces,
            key=lambda x: (
                x.bbox[2] - x.bbox[0]
            ) * (
                x.bbox[3] - x.bbox[1]
            )
        )

        return face.embedding

    # -----------------------------------------------------
    # Compare face embeddings
    # -----------------------------------------------------

    def compare_faces(
        self,
        embedding1,
        embedding2
    ):

        if embedding1 is None or embedding2 is None:
            return 0.0

        embedding1 = torch.tensor(
            embedding1,
            dtype=torch.float32
        )

        embedding2 = torch.tensor(
            embedding2,
            dtype=torch.float32
        )

        similarity = torch.nn.functional.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0)
        ).item()

        return similarity

    def record_voice(
        self,
        filename,
        duration=5,
        sample_rate=16000
    ):
        """
        Record microphone audio.

        Returns:
            Path to saved WAV file.
        """

        print("\nRecording...")
        print(
            f"Speak normally for {duration} seconds."
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
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(
                b"".join(frames)
            )

        print(
            "Recording saved:",
            filename
        )

        return filename
    
    def get_voice_embedding(self, filename):
        """
        Generate ECAPA speaker embedding
        directly from PCM WAV data.

        This avoids TorchCodec / FFmpeg.
        """

        with wave.open(filename, "rb") as wf:

            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frame_count = wf.getnframes()

            audio_bytes = wf.readframes(
                frame_count
            )

        if sample_width != 2:

            raise ValueError(
                "Expected 16-bit audio, "
                f"got {sample_width * 8}-bit"
            )

        if sample_rate != 16000:

            raise ValueError(
                "Expected 16000 Hz audio, "
                f"got {sample_rate} Hz"
            )

        audio = torch.frombuffer(
            audio_bytes,
            dtype=torch.int16
        ).clone()

        # int16 -> float32

        audio = audio.float() / 32768.0

        if channels > 1:

            audio = audio.reshape(
                -1,
                channels
            ).mean(dim=1)

        waveform = audio.unsqueeze(0)

        with torch.no_grad():

            embedding = self.speaker_model.encode_batch(
                waveform
            )

        return embedding.squeeze()

    def compare_voices(
        self,
        embedding1,
        embedding2
    ):
        """
        Compare two speaker embeddings.

        Returns:
            Cosine similarity score.
        """

        if embedding1 is None or embedding2 is None:
            return 0.0

        similarity = torch.nn.functional.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0)
        ).item()

        return similarity

    def verify_identity(
        self,
        face_embedding,
        voice_embedding,
        registered_face_embedding,
        registered_voice_embedding,
        face_threshold=0.50,
        voice_threshold=0.70
    ):
        """
        Verify a student using BOTH face and voice.

        Returns:
            Dictionary containing verification results.
        """

        face_similarity = self.compare_faces(
            face_embedding,
            registered_face_embedding
        )

        voice_similarity = self.compare_voices(
            voice_embedding,
            registered_voice_embedding
        )

        face_match = (
            face_similarity >= face_threshold
        )

        voice_match = (
            voice_similarity >= voice_threshold
        )

        verified = (
            face_match
            and
            voice_match
        )

        return {

            "verified": verified,

            "face_match": face_match,

            "voice_match": voice_match,

            "face_similarity": round(
                face_similarity,
                4
            ),

            "voice_similarity": round(
                voice_similarity,
                4
            )
        }


if __name__ == "__main__":

    engine = RecognitionEngine()

    print("\n================================")
    print("FACE + SPEAKER RECOGNITION TEST")
    print("================================")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print(
            "\nERROR: Camera could not be opened."
        )

        exit()

    print("\nCamera opened.")
    print("Press Q to capture face.")

    captured_face_embedding = None

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "ERROR: Could not read camera frame."
            )

            break

        cv2.imshow(
            "Face Recognition Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            faces = engine.detect_faces(
                frame
            )

            print(
                "\nFaces detected:",
                len(faces)
            )

            if len(faces) > 0:

                face = max(
                    faces,
                    key=lambda x: (
                        x.bbox[2] - x.bbox[0]
                    ) * (
                        x.bbox[3] - x.bbox[1]
                    )
                )

                captured_face_embedding = (
                    face.embedding
                )

                print(
                    "Face captured successfully."
                )

                print(
                    "Embedding size:",
                    len(captured_face_embedding)
                )

            else:

                print(
                    "No face detected."
                )

            break

    camera.release()
    cv2.destroyAllWindows()

    if captured_face_embedding is None:

        print(
            "\nFace capture failed."
        )

        exit()

    print("\nStarting speaker test.")

    voice_file = os.path.join(
        os.path.dirname(__file__),
        "test_voice.wav"
    )

    engine.record_voice(
        filename=voice_file,
        duration=5
    )

    captured_voice_embedding = (
        engine.get_voice_embedding(
            voice_file
        )
    )

    print(
        "\nSpeaker embedding generated."
    )

    print(
        "Embedding size:",
        captured_voice_embedding.shape
    )

    print("\n================================")
    print("CAPTURE SUCCESS")
    print("================================")

    print("Face embedding:  OK")
    print("Voice embedding: OK")

    print(
        "\nThe combined Recognition Engine "
        "is working."
    )