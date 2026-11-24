import sys
import json
import os
import contextlib

import cv2
from insightface.app import FaceAnalysis


def response(data, exit_code=0):
    print(json.dumps(data), flush=True)
    sys.exit(exit_code)


if len(sys.argv) != 7:

    response({
        "success": False,
        "message": "Invalid arguments"
    }, 1)


student_id = sys.argv[1]
name = sys.argv[2]
department = sys.argv[3]
semester = sys.argv[4]
image_path = sys.argv[5]
voice_path = sys.argv[6]


if not os.path.exists(image_path):

    response({
        "success": False,
        "message": "Image file does not exist"
    }, 1)



if not os.path.exists(voice_path):

    response({
        "success": False,
        "message": "Voice file does not exist"
    }, 1)


try:

    semester = int(semester)

except ValueError:

    response({
        "success": False,
        "message": "Semester must be a number"
    }, 1)


try:

    print(
        "Loading face recognition...",
        file=sys.stderr
    )

    with open(os.devnull, "w") as devnull:

        with contextlib.redirect_stdout(devnull):

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
        "message": "Failed to initialize face recognition",
        "error": str(error)
    }, 1)


frame = cv2.imread(image_path)

if frame is None:

    response({
        "success": False,
        "message": "Unable to read captured face image"
    }, 1)


try:

    with open(os.devnull, "w") as devnull:

        with contextlib.redirect_stdout(devnull):

            faces = face_app.get(frame)

except Exception as error:

    response({
        "success": False,
        "message": "Face detection failed",
        "error": str(error)
    }, 1)


if len(faces) == 0:

    response({
        "success": False,
        "message": "No face detected"
    }, 1)


if len(faces) > 1:

    response({
        "success": False,
        "message": "More than one face detected"
    }, 1)



try:

    face_embedding = faces[0].embedding.tolist()

except Exception as error:

    response({
        "success": False,
        "message": "Failed to generate face embedding",
        "error": str(error)
    }, 1)



if len(face_embedding) != 512:

    response({
        "success": False,
        "message": "Invalid face embedding",
        "embedding_dimensions": len(face_embedding)
    }, 1)


response({

    "success": True,

    "message":
        "Face embedding generated successfully",

    "student": {

        "student_id":
            student_id,

        "name":
            name,

        "department":
            department,

        "semester":
            semester

    },

    "authentication": {

        "face":
            True,

        "voice":
            False

    },

    "embedding": {

        "face":
            face_embedding,

        "face_dimensions":
            len(face_embedding),

        "voice_dimensions":
            0

    }

})