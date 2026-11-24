import sys
import json
import os
import contextlib
import subprocess

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from pymongo import MongoClient


PYTHON_EXECUTABLE = sys.executable

FACE_ENGINE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

VOICE_SCRIPT = os.path.join(
    FACE_ENGINE_DIR,
    "voice_embedding.py"
)


def response(data, exit_code=0):

    print(
        json.dumps(data),
        flush=True
    )

    sys.exit(exit_code)

if len(sys.argv) != 3:

    response({

        "success": False,

        "message":
            "Face image path and voice audio path are required."

    }, 1)


image_path = os.path.abspath(
    sys.argv[1]
)

voice_path = os.path.abspath(
    sys.argv[2]
)


if not os.path.isfile(
    image_path
):

    response({

        "success": False,

        "message":
            "Face image file does not exist.",

        "file":
            image_path

    }, 1)


if not os.path.isfile(
    voice_path
):

    response({

        "success": False,

        "message":
            "Voice audio file does not exist.",

        "file":
            voice_path

    }, 1)


if not os.path.isfile(
    VOICE_SCRIPT
):

    response({

        "success": False,

        "message":
            "voice_embedding.py was not found.",

        "file":
            VOICE_SCRIPT

    }, 1)


try:

    print(
        "Generating voice embedding...",
        file=sys.stderr
    )

    voice_process = subprocess.run(

        [

            PYTHON_EXECUTABLE,

            VOICE_SCRIPT,

            voice_path

        ],

        cwd=FACE_ENGINE_DIR,

        stdout=subprocess.PIPE,

        stderr=subprocess.PIPE,

        text=True,

        encoding="utf-8",

        errors="replace"

    )


except Exception as error:

    response({

        "success": False,

        "message":
            "Failed to start voice embedding process.",

        "error":
            str(error)

    }, 1)



if voice_process.returncode != 0:

    response({

        "success": False,

        "message":
            "Voice embedding generation failed.",

        "error":
            voice_process.stderr.strip(),

        "voice_output":
            voice_process.stdout.strip()

    }, 1)


try:

    voice_output = (
        voice_process.stdout.strip()
    )

    voice_data = json.loads(
        voice_output
    )


except Exception as error:

    response({

        "success": False,

        "message":
            "Invalid voice embedding response.",

        "error":
            str(error),

        "voice_output":
            voice_process.stdout.strip()

    }, 1)


if not voice_data.get(
    "success",
    False
):

    response({

        "success": False,

        "message":
            "Voice embedding generation was unsuccessful.",

        "voice_response":
            voice_data

    }, 1)


query_voice_embedding = (
    voice_data.get(
        "embedding"
    )
)


if not isinstance(
    query_voice_embedding,
    list
):

    response({

        "success": False,

        "message":
            "Voice embedding is missing or invalid."

    }, 1)


if len(
    query_voice_embedding
) != 192:

    response({

        "success": False,

        "message":
            "Invalid voice embedding dimensions.",

        "dimensions":
            len(
                query_voice_embedding
            )

    }, 1)


try:

    client = MongoClient(

        "mongodb://localhost:27017/",

        serverSelectionTimeoutMS=5000

    )

    client.admin.command(
        "ping"
    )

    db = client[
        "StudentDB"
    ]

    students = db[
        "students"
    ]


except Exception as error:

    response({

        "success": False,

        "message":
            "MongoDB connection failed.",

        "error":
            str(error)

    }, 1)


frame = cv2.imread(
    image_path
)


if frame is None:

    response({

        "success": False,

        "message":
            "Unable to read face image."

    }, 1)



try:

    print(
        "Loading face recognition...",
        file=sys.stderr
    )

    with open(
        os.devnull,
        "w"
    ) as devnull:

        with contextlib.redirect_stdout(
            devnull
        ):

            face_app = FaceAnalysis(

                name="buffalo_l"

            )

            face_app.prepare(

                ctx_id=0,

                det_size=(640, 640)

            )


except Exception as error:

    response({

        "success": False,

        "message":
            "Failed to initialize face recognition.",

        "error":
            str(error)

    }, 1)


try:

    with open(
        os.devnull,
        "w"
    ) as devnull:

        with contextlib.redirect_stdout(
            devnull
        ):

            faces = face_app.get(
                frame
            )


except Exception as error:

    response({

        "success": False,

        "message":
            "Face detection failed.",

        "error":
            str(error)

    }, 1)


if len(faces) == 0:

    response({

        "success": True,

        "recognized": False,

        "face_verified": False,

        "voice_verified": False,

        "message":
            "No face detected."

    })


if len(faces) > 1:

    response({

        "success": True,

        "recognized": False,

        "face_verified": False,

        "voice_verified": False,

        "message":
            "More than one face detected."

    })


try:

    query_face_embedding = (
        faces[0].embedding
    )

except Exception as error:

    response({

        "success": False,

        "message":
            "Failed to generate face embedding.",

        "error":
            str(error)

    }, 1)


if (

    query_face_embedding is None

    or

    len(query_face_embedding) != 512

):

    response({

        "success": False,

        "recognized": False,

        "message":
            "Invalid face embedding."

    }, 1)



def cosine_similarity(a, b):

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    if a.shape != b.shape:

        return 0.0

    denominator = (

        np.linalg.norm(a)

        *

        np.linalg.norm(b)

    )

    if denominator == 0:

        return 0.0

    return float(

        np.dot(a, b)

        /

        denominator

    )


try:

    students_list = list(
        students.find({})
    )

except Exception as error:

    response({

        "success": False,

        "message":
            "Failed to read students.",

        "error":
            str(error)

    }, 1)



if len(
    students_list
) == 0:

    response({

        "success": True,

        "recognized": False,

        "message":
            "No registered students found."

    })


best_student = None

best_face_score = -1.0

best_voice_score = -1.0


for student in students_list:
    stored_face_embedding = (
        student.get(
            "face_embedding"
        )
    )

    if not stored_face_embedding:

        continue

    if len(
        stored_face_embedding
    ) != 512:

        continue


    stored_voice_embedding = (
        student.get(
            "voice_embedding"
        )
    )

    if not stored_voice_embedding:

        continue

    if len(
        stored_voice_embedding
    ) != 192:

        continue


    face_score = cosine_similarity(

        query_face_embedding,

        stored_face_embedding

    )


    voice_score = cosine_similarity(

        query_voice_embedding,

        stored_voice_embedding

    )


    combined_score = (

        face_score

        +

        voice_score

    ) / 2.0


    best_combined_score = (

        (
            best_face_score
            +
            best_voice_score
        ) / 2.0

        if best_student is not None

        else -1.0

    )


    if combined_score > best_combined_score:

        best_student = student

        best_face_score = face_score

        best_voice_score = voice_score


FACE_THRESHOLD = 0.45

VOICE_THRESHOLD = 0.75


if best_student is None:

    response({

        "success": True,

        "recognized": False,

        "face_verified": False,

        "voice_verified": False,

        "message":
            "No registered face and voice match found."

    })

face_verified = (

    best_face_score

    >=

    FACE_THRESHOLD

)


voice_verified = (

    best_voice_score

    >=

    VOICE_THRESHOLD

)


if not face_verified or not voice_verified:

    response({

        "success": True,

        "recognized": False,

        "face_verified":
            face_verified,

        "voice_verified":
            voice_verified,

        "message":
            "Face and voice authentication failed.",

        "face_similarity":
            round(
                best_face_score,
                4
            ),

        "voice_similarity":
            round(
                best_voice_score,
                4
            )

    })


response({

    "success": True,

    "recognized": True,

    "face_verified": True,

    "voice_verified": True,

    "message":
        "Face and voice recognized successfully.",

    "student": {

        "student_id":
            str(
                best_student[
                    "student_id"
                ]
            ),

        "name":
            best_student[
                "name"
            ],

        "department":
            best_student[
                "department"
            ],

        "semester":
            best_student[
                "semester"
            ]

    },

    "face_similarity":
        round(
            best_face_score,
            4
        ),

    "voice_similarity":
        round(
            best_voice_score,
            4
        )

})