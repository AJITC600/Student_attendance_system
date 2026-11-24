const express = require("express");
const cors = require("cors");
const multer = require("multer");
const { MongoClient } = require("mongodb");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const app = express();

app.use(cors());
app.use(express.json());

const upload = multer({
    storage: multer.memoryStorage()
});

const MONGO_URI = "mongodb://localhost:27017/";
const client = new MongoClient(MONGO_URI);

let studentsCollection = null;
let loginHistoryCollection = null;

const VOICE_THRESHOLD = 0.75;

const PYTHON_EXECUTABLE =
    process.env.PYTHON_EXECUTABLE ||
    "C:\\Users\\HP\\anaconda3\\envs\\student_attend\\python.exe";

const tempDirectory = path.join(__dirname, "temp");
const faceEngineDirectory = path.join(__dirname, "..", "face_engine");

const registerScript = path.join(
    faceEngineDirectory,
    "register.py"
);

const recognizeScript = path.join(
    faceEngineDirectory,
    "recognize.py"
);

const voiceScript = path.join(
    faceEngineDirectory,
    "voice_embedding.py"
);

if (!fs.existsSync(tempDirectory)) {
    fs.mkdirSync(tempDirectory, {
        recursive: true
    });
}

console.log("\n========================================");
console.log("PYTHON CONFIGURATION");
console.log("========================================");

console.log("Python:", PYTHON_EXECUTABLE);
console.log("Python exists:", fs.existsSync(PYTHON_EXECUTABLE));

console.log("Face engine:", faceEngineDirectory);
console.log(
    "Face engine exists:",
    fs.existsSync(faceEngineDirectory)
);

console.log("register.py:", registerScript);
console.log(
    "register.py exists:",
    fs.existsSync(registerScript)
);

console.log("recognize.py:", recognizeScript);
console.log(
    "recognize.py exists:",
    fs.existsSync(recognizeScript)
);

console.log("voice_embedding.py:", voiceScript);
console.log(
    "voice_embedding.py exists:",
    fs.existsSync(voiceScript)
);

console.log("========================================\n");

app.get("/", (req, res) => {
    return res.json({
        success: true,
        message: "Student Face + Voice Recognition API is running!"
    });
});

app.post(
    "/register",
    upload.fields([
        {
            name: "face",
            maxCount: 1
        },
        {
            name: "voice",
            maxCount: 1
        }
    ]),
    async (req, res) => {
        let imagePath = null;
        let voicePath = null;
        let studentCreatedByRegisterPy = false;

        try {
            console.log("\n========================================");
            console.log("STUDENT FACE + VOICE REGISTRATION");
            console.log("========================================");

            const {
                student_id,
                name,
                department,
                semester
            } = req.body;

            const face = req.files?.face?.[0];
            const voice = req.files?.voice?.[0];

            if (
                !student_id ||
                !name ||
                !department ||
                !semester
            ) {
                return res.status(400).json({
                    success: false,
                    message: "Student details are required"
                });
            }

            if (!face) {
                return res.status(400).json({
                    success: false,
                    message: "Face image is required"
                });
            }

            if (!voice) {
                return res.status(400).json({
                    success: false,
                    message: "Voice recording is required"
                });
            }

            const studentId = String(student_id).trim();

            const existingStudent =
                await studentsCollection.findOne({
                    $or: [
                        {
                            _id: studentId
                        },
                        {
                            student_id: studentId
                        }
                    ]
                });

            if (existingStudent) {
                return res.status(409).json({
                    success: false,
                    message: "Student ID already exists"
                });
            }

            const timestamp = Date.now();

            imagePath = path.join(
                tempDirectory,
                `${studentId}_${timestamp}.jpg`
            );

            const voiceExtension =
                path.extname(voice.originalname) || ".webm";

            voicePath = path.join(
                tempDirectory,
                `${studentId}_${timestamp}${voiceExtension}`
            );

            fs.writeFileSync(
                imagePath,
                face.buffer
            );

            fs.writeFileSync(
                voicePath,
                voice.buffer
            );

            console.log(
                "\n[1/2] Starting face registration..."
            );

            const faceResult =
                await runPython([
                    registerScript,
                    studentId,
                    String(name),
                    String(department),
                    String(semester),
                    imagePath,
                    voicePath
                ]);

            if (faceResult.code !== 0) {
                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(500).json({
                    success: false,
                    message: "Face registration failed",
                    error: faceResult.stderr
                });
            }

            let faceData;

            try {
                faceData =
                    parsePythonJSON(
                        faceResult.stdout
                    );
            } catch (error) {
                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(500).json({
                    success: false,
                    message:
                        "Invalid response from face recognition system",
                    error: error.message
                });
            }

            if (!faceData.success) {
                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(400).json({
                    success: false,
                    message:
                        faceData.message ||
                        "Face registration failed"
                });
            }

            studentCreatedByRegisterPy = true;

            let faceEmbedding =
                faceData.embedding?.face;

            if (!faceEmbedding) {
                await removeIncompleteStudent(
                    studentId
                );

                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(400).json({
                    success: false,
                    message:
                        "Face registration succeeded, but face embedding was not returned"
                });
            }

            if (!Array.isArray(faceEmbedding)) {
                faceEmbedding =
                    Array.from(faceEmbedding);
            }

            if (faceEmbedding.length !== 512) {
                await removeIncompleteStudent(
                    studentId
                );

                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(400).json({
                    success: false,
                    message: "Invalid face embedding",
                    face_embedding_size:
                        faceEmbedding.length
                });
            }

            console.log(
                "Face embedding size:",
                faceEmbedding.length
            );

            console.log(
                "\n[2/2] Generating voice embedding..."
            );

            const voiceResult =
                await runPython([
                    voiceScript,
                    voicePath
                ]);

            if (voiceResult.code !== 0) {
                await removeIncompleteStudent(
                    studentId
                );

                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(500).json({
                    success: false,
                    message: "Voice registration failed",
                    error: voiceResult.stderr
                });
            }

            let voiceData;

            try {
                voiceData =
                    parsePythonJSON(
                        voiceResult.stdout
                    );
            } catch (error) {
                await removeIncompleteStudent(
                    studentId
                );

                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(500).json({
                    success: false,
                    message:
                        "Invalid response from voice recognition system",
                    error: error.message
                });
            }

            if (
                !voiceData.success ||
                !voiceData.embedding
            ) {
                await removeIncompleteStudent(
                    studentId
                );

                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(400).json({
                    success: false,
                    message:
                        voiceData.message ||
                        "Voice embedding generation failed"
                });
            }

            let voiceEmbedding =
                voiceData.embedding;

            if (!Array.isArray(voiceEmbedding)) {
                voiceEmbedding =
                    Array.from(voiceEmbedding);
            }

            if (voiceEmbedding.length !== 192) {
                await removeIncompleteStudent(
                    studentId
                );

                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(400).json({
                    success: false,
                    message: "Invalid voice embedding",
                    voice_embedding_size:
                        voiceEmbedding.length
                });
            }

            console.log(
                "Voice embedding size:",
                voiceEmbedding.length
            );

            const now = new Date();

            const studentDocument = {
                student_id: studentId,
                name: String(name),
                department: String(department),
                semester: Number(semester),

                face_embedding: faceEmbedding,
                voice_embedding: voiceEmbedding,

                face_registered: true,
                voice_registered: true,

                face_registered_at: now,
                voice_registered_at: now,

                updated_at: now
            };

            const updateResult =
                await studentsCollection.updateOne(
                    {
                        $or: [
                            {
                                _id: studentId
                            },
                            {
                                student_id: studentId
                            }
                        ]
                    },
                    {
                        $set: studentDocument,
                        $setOnInsert: {
                            _id: studentId,
                            created_at: now
                        }
                    },
                    {
                        upsert: true
                    }
                );

            console.log(
                "Matched:",
                updateResult.matchedCount
            );

            console.log(
                "Modified:",
                updateResult.modifiedCount
            );

            console.log(
                "Upserted:",
                updateResult.upsertedCount
            );

            const savedStudent =
                await studentsCollection.findOne({
                    $or: [
                        {
                            student_id: studentId
                        },
                        {
                            _id: studentId
                        }
                    ]
                });

            if (!savedStudent) {
                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(500).json({
                    success: false,
                    message:
                        "Student registration completed but MongoDB verification failed"
                });
            }

            if (
                !Array.isArray(
                    savedStudent.face_embedding
                ) ||
                savedStudent.face_embedding.length !== 512
            ) {
                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(500).json({
                    success: false,
                    message:
                        "Face embedding was not correctly saved"
                });
            }

            if (
                !Array.isArray(
                    savedStudent.voice_embedding
                ) ||
                savedStudent.voice_embedding.length !== 192
            ) {
                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(500).json({
                    success: false,
                    message:
                        "Voice embedding was not correctly saved"
                });
            }

            cleanupFiles(
                imagePath,
                voicePath
            );

            imagePath = null;
            voicePath = null;

            console.log(
                "\n========================================"
            );

            console.log(
                "REGISTRATION SUCCESSFUL"
            );

            console.log(
                "Student ID:",
                studentId
            );

            console.log(
                "Face:",
                faceEmbedding.length
            );

            console.log(
                "Voice:",
                voiceEmbedding.length
            );

            console.log(
                "========================================\n"
            );

            return res.json({
                success: true,
                message:
                    "Student registered successfully with face and voice",

                student: {
                    student_id: studentId,
                    name: name,
                    department: department,
                    semester: Number(semester)
                },

                face: {
                    registered: true,
                    embedding_size:
                        faceEmbedding.length
                },

                voice: {
                    registered: true,
                    embedding_size:
                        voiceEmbedding.length
                }
            });
        }

        catch (error) {
            console.error(
                "Registration error:",
                error
            );

            if (
                studentCreatedByRegisterPy &&
                req.body?.student_id
            ) {
                await removeIncompleteStudent(
                    String(req.body.student_id)
                );
            }

            cleanupFiles(
                imagePath,
                voicePath
            );

            return res.status(500).json({
                success: false,
                message: "Registration failed",
                error: error.message
            });
        }
    }
);

app.post(
    "/recognize",
    upload.fields([
        {
            name: "face",
            maxCount: 1
        },
        {
            name: "voice",
            maxCount: 1
        }
    ]),
    async (req, res) => {
        let imagePath = null;
        let voicePath = null;

        try {
            console.log("\n========================================");
            console.log("FACE + VOICE LOGIN");
            console.log("========================================");

            const face =
                req.files?.face?.[0];

            const voice =
                req.files?.voice?.[0];

            if (!face) {
                return res.status(400).json({
                    success: false,
                    message: "Face image is required"
                });
            }

            if (!voice) {
                return res.status(400).json({
                    success: false,
                    message:
                        "Voice recording is required"
                });
            }

            const timestamp = Date.now();

            imagePath = path.join(
                tempDirectory,
                `recognize_${timestamp}.jpg`
            );

            const voiceExtension =
                path.extname(
                    voice.originalname
                ) || ".webm";

            voicePath = path.join(
                tempDirectory,
                `verify_${timestamp}${voiceExtension}`
            );

            fs.writeFileSync(
                imagePath,
                face.buffer
            );

            fs.writeFileSync(
                voicePath,
                voice.buffer
            );

            console.log(
                "\n[1/2] Running face recognition..."
            );

            const faceResult =
                await runPython([
                    recognizeScript,
                    imagePath,
                    voicePath
                ]);

            if (faceResult.code !== 0) {
                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(500).json({
                    success: false,
                    message:
                        "Face recognition failed",
                    error:
                        faceResult.stderr
                });
            }

            let faceData;

            try {
                faceData =
                    parsePythonJSON(
                        faceResult.stdout
                    );
            } catch (error) {
                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(500).json({
                    success: false,
                    message:
                        "Invalid response from recognition system",
                    error:
                        error.message
                });
            }

            if (!faceData.recognized) {
                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.json({
                    ...faceData,
                    verified: false,
                    faceVerified: false,
                    voiceVerified: false
                });
            }

            const student =
                faceData.student;

            if (!student) {
                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(500).json({
                    success: false,
                    recognized: true,
                    verified: false,
                    message:
                        "Face recognized but student information is missing"
                });
            }

            const studentId =
                String(
                    student.student_id
                );

            console.log(
                "Face recognized:",
                student.name
            );

            console.log(
                "Student ID:",
                studentId
            );

            const registeredStudent =
                await studentsCollection.findOne({
                    $or: [
                        {
                            student_id: studentId
                        },
                        {
                            _id: studentId
                        }
                    ]
                });

            if (!registeredStudent) {
                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(404).json({
                    success: false,
                    recognized: true,
                    verified: false,
                    message:
                        "Student exists in face system but was not found in StudentDB"
                });
            }

            if (
                !Array.isArray(
                    registeredStudent.voice_embedding
                ) ||
                registeredStudent.voice_embedding.length === 0
            ) {
                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(400).json({
                    success: false,
                    recognized: true,
                    verified: false,
                    faceVerified: true,
                    voiceVerified: false,
                    message:
                        "Voice is not registered for this student"
                });
            }

            console.log(
                "\n[2/2] Generating test voice embedding..."
            );

            const voiceResult =
                await runPython([
                    voiceScript,
                    voicePath
                ]);

            if (voiceResult.code !== 0) {
                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(500).json({
                    success: false,
                    recognized: true,
                    verified: false,
                    faceVerified: true,
                    voiceVerified: false,
                    message:
                        "Voice processing failed",
                    error:
                        voiceResult.stderr
                });
            }

            let voiceData;

            try {
                voiceData =
                    parsePythonJSON(
                        voiceResult.stdout
                    );
            } catch (error) {
                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(500).json({
                    success: false,
                    recognized: true,
                    verified: false,
                    faceVerified: true,
                    voiceVerified: false,
                    message:
                        "Invalid response from voice recognition system",
                    error:
                        error.message
                });
            }

            if (
                !voiceData.success ||
                !voiceData.embedding
            ) {
                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(400).json({
                    success: false,
                    recognized: true,
                    verified: false,
                    faceVerified: true,
                    voiceVerified: false,
                    message:
                        voiceData.message ||
                        "Could not generate voice embedding"
                });
            }

            let testVoiceEmbedding =
                voiceData.embedding;

            if (!Array.isArray(testVoiceEmbedding)) {
                testVoiceEmbedding =
                    Array.from(
                        testVoiceEmbedding
                    );
            }

            const registeredEmbedding =
                registeredStudent.voice_embedding;

            if (
                registeredEmbedding.length !==
                testVoiceEmbedding.length
            ) {
                cleanupFiles(
                    imagePath,
                    voicePath
                );

                return res.status(500).json({
                    success: false,
                    recognized: true,
                    verified: false,
                    faceVerified: true,
                    voiceVerified: false,
                    message:
                        "Voice embedding dimensions do not match"
                });
            }

            const voiceSimilarity =
                cosineSimilarity(
                    registeredEmbedding,
                    testVoiceEmbedding
                );

            const voiceVerified =
                voiceSimilarity >=
                VOICE_THRESHOLD;

            console.log(
                "Voice similarity:",
                voiceSimilarity
            );

            console.log(
                "Voice verified:",
                voiceVerified
            );

            cleanupFiles(
                imagePath,
                voicePath
            );

            imagePath = null;
            voicePath = null;

            if (!voiceVerified) {
                return res.json({
                    success: true,
                    recognized: true,
                    verified: false,
                    faceVerified: true,
                    voiceVerified: false,
                    action:
                        "verification_failed",
                    message:
                        "Face recognized, but voice verification failed.",
                    student: {
                        student_id:
                            registeredStudent.student_id,
                        name:
                            registeredStudent.name,
                        department:
                            registeredStudent.department,
                        semester:
                            registeredStudent.semester
                    },
                    faceSimilarity:
                        faceData.similarity,
                    voiceSimilarity:
                        Number(
                            voiceSimilarity.toFixed(4)
                        ),
                    voiceThreshold:
                        VOICE_THRESHOLD
                });
            }

            console.log(
                "Face + Voice verification successful."
            );

            const now = new Date();

            const today =
                getLocalDate(now);

            const currentTime =
                getLocalTime(now);

            const todayLogin =
                await loginHistoryCollection.findOne({
                    student_id: studentId,
                    login_date: today
                });

            if (!todayLogin) {
                try {
                    const loginRecord = {
                        student_id: studentId,
                        name:
                            registeredStudent.name,
                        department:
                            registeredStudent.department,
                        semester:
                            registeredStudent.semester,
                        login_date: today,
                        login_time: currentTime,
                        logout_time: null,
                        status: "logged_in"
                    };

                    await loginHistoryCollection.insertOne(
                        loginRecord
                    );

                    console.log(
                        `Student ${studentId} logged in.`
                    );

                    return res.json({
                        success: true,
                        recognized: true,
                        verified: true,
                        faceVerified: true,
                        voiceVerified: true,
                        login: true,
                        alreadyLoggedIn: false,
                        locked: false,
                        action: "first_login",
                        message:
                            "Face and voice verified. Login successful.",
                        student: {
                            student_id:
                                registeredStudent.student_id,
                            name:
                                registeredStudent.name,
                            department:
                                registeredStudent.department,
                            semester:
                                registeredStudent.semester
                        },
                        faceSimilarity:
                            faceData.similarity,
                        voiceSimilarity:
                            Number(
                                voiceSimilarity.toFixed(4)
                            ),
                        voiceThreshold:
                            VOICE_THRESHOLD
                    });
                }

                catch (error) {
                    if (error.code !== 11000) {
                        throw error;
                    }

                    console.log(
                        `Login record already exists for ${studentId}.`
                    );
                }
            }

            const currentLogin =
                await loginHistoryCollection.findOne({
                    student_id: studentId,
                    login_date: today
                });

            if (!currentLogin) {
                return res.status(500).json({
                    success: false,
                    recognized: true,
                    verified: true,
                    message:
                        "Unable to determine this student's login status"
                });
            }

            if (
                currentLogin.status ===
                "logged_in"
            ) {
                console.log(
                    `Student ${studentId} is already logged in.`
                );

                return res.json({
                    success: true,
                    recognized: true,
                    verified: true,
                    faceVerified: true,
                    voiceVerified: true,

                    login: false,
                    alreadyLoggedIn: true,
                    locked: false,

                    action:
                        "logout_confirmation",

                    message:
                        "You are already logged in. Do you want to logout?",

                    student: {
                        student_id:
                            registeredStudent.student_id,
                        name:
                            registeredStudent.name,
                        department:
                            registeredStudent.department,
                        semester:
                            registeredStudent.semester
                    },

                    student_id:
                        studentId,

                    faceSimilarity:
                        faceData.similarity,

                    voiceSimilarity:
                        Number(
                            voiceSimilarity.toFixed(4)
                        ),

                    voiceThreshold:
                        VOICE_THRESHOLD
                });
            }

            if (
                currentLogin.status ===
                "logged_out"
            ) {
                console.log(
                    `Student ${studentId} is locked for today.`
                );

                return res.json({
                    success: true,
                    recognized: true,
                    verified: true,
                    faceVerified: true,
                    voiceVerified: true,

                    login: false,
                    alreadyLoggedIn: false,
                    locked: true,

                    action: "locked",

                    message:
                        "This student is locked for today. The student can login again tomorrow.",

                    student: {
                        student_id:
                            registeredStudent.student_id,
                        name:
                            registeredStudent.name,
                        department:
                            registeredStudent.department,
                        semester:
                            registeredStudent.semester
                    },

                    student_id:
                        studentId,

                    faceSimilarity:
                        faceData.similarity,

                    voiceSimilarity:
                        Number(
                            voiceSimilarity.toFixed(4)
                        ),

                    voiceThreshold:
                        VOICE_THRESHOLD
                });
            }

            return res.json({
                success: false,
                recognized: true,
                verified: true,
                student_id: studentId,
                message:
                    "Unknown login status"
            });
        }

        catch (error) {
            console.error(
                "Recognition error:",
                error
            );

            cleanupFiles(
                imagePath,
                voicePath
            );

            return res.status(500).json({
                success: false,
                message:
                    "Face + voice recognition failed",
                error:
                    error.message
            });
        }
    }
);

app.post(
    "/logout",
    async (req, res) => {
        try {
            const {
                student_id
            } = req.body;

            if (!student_id) {
                return res.status(400).json({
                    success: false,
                    message:
                        "Student ID is required"
                });
            }

            const studentId =
                String(student_id).trim();

            const student =
                await studentsCollection.findOne({
                    $or: [
                        {
                            student_id:
                                studentId
                        },
                        {
                            _id:
                                studentId
                        }
                    ]
                });

            if (!student) {
                return res.status(404).json({
                    success: false,
                    message:
                        "Student not found",
                    student_id:
                        studentId
                });
            }

            const now = new Date();

            const today =
                getLocalDate(now);

            const currentTime =
                getLocalTime(now);

            const result =
                await loginHistoryCollection.updateOne(
                    {
                        student_id:
                            studentId,
                        login_date:
                            today,
                        status:
                            "logged_in"
                    },
                    {
                        $set: {
                            logout_time:
                                currentTime,
                            status:
                                "logged_out"
                        }
                    }
                );

            if (
                result.matchedCount === 0
            ) {
                const existingLogin =
                    await loginHistoryCollection.findOne({
                        student_id:
                            studentId,
                        login_date:
                            today
                    });

                if (
                    existingLogin?.status ===
                    "logged_out"
                ) {
                    return res.json({
                        success: true,
                        student_id:
                            studentId,
                        locked: true,
                        action:
                            "locked",
                        message:
                            "This student is already logged out and locked for today."
                    });
                }

                return res.json({
                    success: false,
                    student_id:
                        studentId,
                    locked: false,
                    message:
                        "No active login found for this student today"
                });
            }

            console.log(
                `Student ${studentId} logged out.`
            );

            return res.json({
                success: true,
                student_id:
                    studentId,
                locked: true,
                action:
                    "locked",
                message:
                    "Logout successful. This student is locked for today."
            });
        }

        catch (error) {
            console.error(
                "Logout error:",
                error
            );

            return res.status(500).json({
                success: false,
                message:
                    "Logout failed",
                error:
                    error.message
            });
        }
    }
);

app.post(
    "/voice/verify",
    upload.single("voice"),
    async (req, res) => {
        let voicePath = null;

        try {
            const {
                student_id
            } = req.body;

            if (!student_id) {
                return res.status(400).json({
                    success: false,
                    message:
                        "Student ID is required"
                });
            }

            if (!req.file) {
                return res.status(400).json({
                    success: false,
                    message:
                        "Voice recording is required"
                });
            }

            const studentId =
                String(student_id).trim();

            const student =
                await studentsCollection.findOne({
                    $or: [
                        {
                            student_id:
                                studentId
                        },
                        {
                            _id:
                                studentId
                        }
                    ]
                });

            if (!student) {
                return res.status(404).json({
                    success: false,
                    message:
                        "Student not found"
                });
            }

            if (
                !Array.isArray(
                    student.voice_embedding
                ) ||
                student.voice_embedding.length === 0
            ) {
                return res.status(400).json({
                    success: false,
                    message:
                        "Voice is not registered for this student"
                });
            }

            const extension =
                path.extname(
                    req.file.originalname
                ) || ".webm";

            voicePath = path.join(
                tempDirectory,
                `voice_test_${Date.now()}${extension}`
            );

            fs.writeFileSync(
                voicePath,
                req.file.buffer
            );

            const result =
                await runPython([
                    voiceScript,
                    voicePath
                ]);

            cleanupFiles(
                voicePath
            );

            voicePath = null;

            if (result.code !== 0) {
                return res.status(500).json({
                    success: false,
                    message:
                        "Voice processing failed",
                    error:
                        result.stderr
                });
            }

            let data;

            try {
                data =
                    parsePythonJSON(
                        result.stdout
                    );
            } catch (error) {
                return res.status(500).json({
                    success: false,
                    message:
                        "Invalid response from voice recognition system",
                    error:
                        error.message
                });
            }

            if (
                !data.success ||
                !data.embedding
            ) {
                return res.status(400).json({
                    success: false,
                    message:
                        data.message ||
                        "Could not generate voice embedding"
                });
            }

            const testEmbedding =
                Array.isArray(data.embedding)
                    ? data.embedding
                    : Array.from(
                        data.embedding
                    );

            const similarity =
                cosineSimilarity(
                    student.voice_embedding,
                    testEmbedding
                );

            const verified =
                similarity >=
                VOICE_THRESHOLD;

            return res.json({
                success: true,
                student_id:
                    studentId,
                verified:
                    verified,
                similarity:
                    Number(
                        similarity.toFixed(4)
                    ),
                threshold:
                    VOICE_THRESHOLD,
                message:
                    verified
                        ? "Voice verified successfully."
                        : "Voice verification failed."
            });
        }

        catch (error) {
            cleanupFiles(
                voicePath
            );

            console.error(
                "Voice verification error:",
                error
            );

            return res.status(500).json({
                success: false,
                message:
                    "Voice verification failed",
                error:
                    error.message
            });
        }
    }
);

async function removeIncompleteStudent(studentId) {
    try {
        const id =
            String(studentId).trim();

        const result =
            await studentsCollection.deleteOne({
                $or: [
                    {
                        _id: id
                    },
                    {
                        student_id: id
                    }
                ]
            });

        if (result.deletedCount > 0) {
            console.log(
                `Removed incomplete registration for student ${id}`
            );
        }
    }

    catch (error) {
        console.error(
            "Could not remove incomplete student:",
            error.message
        );
    }
}

function cosineSimilarity(a, b) {
    if (
        !Array.isArray(a) ||
        !Array.isArray(b)
    ) {
        throw new Error(
            "Embeddings must be arrays"
        );
    }

    if (
        a.length !== b.length
    ) {
        throw new Error(
            "Embedding dimensions do not match"
        );
    }

    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;

    for (
        let i = 0;
        i < a.length;
        i++
    ) {
        const x =
            Number(a[i]);

        const y =
            Number(b[i]);

        if (
            !Number.isFinite(x) ||
            !Number.isFinite(y)
        ) {
            throw new Error(
                "Embedding contains invalid values"
            );
        }

        dotProduct +=
            x * y;

        magnitudeA +=
            x * x;

        magnitudeB +=
            y * y;
    }

    magnitudeA =
        Math.sqrt(
            magnitudeA
        );

    magnitudeB =
        Math.sqrt(
            magnitudeB
        );

    if (
        magnitudeA === 0 ||
        magnitudeB === 0
    ) {
        return 0;
    }

    return (
        dotProduct /
        (
            magnitudeA *
            magnitudeB
        )
    );
}

function parsePythonJSON(output) {
    if (
        typeof output !== "string"
    ) {
        throw new Error(
            "Python output is not text"
        );
    }

    const text =
        output.trim();

    if (!text) {
        throw new Error(
            "Python returned empty output"
        );
    }

    try {
        return JSON.parse(
            text
        );
    }

    catch (error) {
    }

    const firstBrace =
        text.indexOf("{");

    const lastBrace =
        text.lastIndexOf("}");

    if (
        firstBrace === -1 ||
        lastBrace === -1 ||
        lastBrace <= firstBrace
    ) {
        throw new Error(
            "Python did not return valid JSON"
        );
    }

    const jsonText =
        text.substring(
            firstBrace,
            lastBrace + 1
        );

    return JSON.parse(
        jsonText
    );
}

function runPython(args) {
    return new Promise(
        (resolve) => {
            console.log(
                "\n----------------------------------------"
            );

            console.log(
                "Starting Python:"
            );

            console.log(
                PYTHON_EXECUTABLE
            );

            console.log(
                "Working directory:"
            );

            console.log(
                faceEngineDirectory
            );

            console.log(
                "Arguments:"
            );

            console.log(
                args
            );

            console.log(
                "----------------------------------------"
            );

            const pythonProcess =
                spawn(
                    PYTHON_EXECUTABLE,
                    args,
                    {
                        cwd:
                            faceEngineDirectory,
                        windowsHide:
                            true
                    }
                );

            let stdout = "";
            let stderr = "";

            pythonProcess.stdout.on(
                "data",
                (data) => {
                    stdout +=
                        data.toString();
                }
            );

            pythonProcess.stderr.on(
                "data",
                (data) => {
                    stderr +=
                        data.toString();
                }
            );

            pythonProcess.on(
                "error",
                (error) => {
                    console.error(
                        "Python process error:",
                        error
                    );

                    resolve({
                        code: 1,
                        stdout:
                            stdout,
                        stderr:
                            error.message
                    });
                }
            );

            pythonProcess.on(
                "close",
                (code) => {
                    console.log(
                        "Python process finished."
                    );

                    console.log(
                        "Exit code:",
                        code
                    );

                    if (stdout.trim()) {
                        console.log(
                            "\nPython stdout:"
                        );

                        console.log(
                            stdout
                        );
                    }

                    if (stderr.trim()) {
                        console.error(
                            "\nPython stderr:"
                        );

                        console.error(
                            stderr
                        );
                    }

                    resolve({
                        code:
                            code,
                        stdout:
                            stdout,
                        stderr:
                            stderr
                    });
                }
            );
        }
    );
}

function cleanupFiles(...files) {
    for (
        const file of files
    ) {
        try {
            if (
                file &&
                fs.existsSync(file)
            ) {
                fs.unlinkSync(
                    file
                );
            }
        }

        catch (error) {
            console.error(
                "Could not delete:",
                file,
                error.message
            );
        }
    }
}

function getLocalDate(date) {
    const year =
        date.getFullYear();

    const month =
        String(
            date.getMonth() + 1
        ).padStart(
            2,
            "0"
        );

    const day =
        String(
            date.getDate()
        ).padStart(
            2,
            "0"
        );

    return `${year}-${month}-${day}`;
}

function getLocalTime(date) {
    const hours =
        String(
            date.getHours()
        ).padStart(
            2,
            "0"
        );

    const minutes =
        String(
            date.getMinutes()
        ).padStart(
            2,
            "0"
        );

    const seconds =
        String(
            date.getSeconds()
        ).padStart(
            2,
            "0"
        );

    return `${hours}:${minutes}:${seconds}`;
}

async function startServer() {
    try {
        if (
            !fs.existsSync(
                PYTHON_EXECUTABLE
            )
        ) {
            throw new Error(
                `Python executable not found:\n${PYTHON_EXECUTABLE}`
            );
        }

        if (
            !fs.existsSync(
                faceEngineDirectory
            )
        ) {
            throw new Error(
                `Face engine directory not found:\n${faceEngineDirectory}`
            );
        }

        if (
            !fs.existsSync(
                registerScript
            )
        ) {
            throw new Error(
                `register.py not found:\n${registerScript}`
            );
        }

        if (
            !fs.existsSync(
                recognizeScript
            )
        ) {
            throw new Error(
                `recognize.py not found:\n${recognizeScript}`
            );
        }

        if (
            !fs.existsSync(
                voiceScript
            )
        ) {
            throw new Error(
                `voice_embedding.py not found:\n${voiceScript}`
            );
        }

        await client.connect();

        console.log(
            "MongoDB connected successfully!"
        );

        const studentDB =
            client.db(
                "StudentDB"
            );

        studentsCollection =
            studentDB.collection(
                "students"
            );

        const loginDB =
            client.db(
                "StudentLoginDB"
            );

        loginHistoryCollection =
            loginDB.collection(
                "login_history"
            );

        console.log(
            "StudentDB connected"
        );

        console.log(
            "StudentLoginDB connected"
        );

        try {
            await studentsCollection.createIndex(
                {
                    student_id: 1
                },
                {
                    unique: true
                }
            );
        }

        catch (error) {
            console.log(
                "Student ID index:",
                error.message
            );
        }

        try {
            await loginHistoryCollection.createIndex(
                {
                    student_id: 1,
                    login_date: 1
                },
                {
                    unique: true
                }
            );

            console.log(
                "Login history unique index created."
            );
        }

        catch (error) {
            console.log(
                "Login history index:",
                error.message
            );
        }

        app.listen(
            5000,
            () => {
                console.log(
                    "\n========================================"
                );

                console.log(
                    "SERVER STARTED"
                );

                console.log(
                    "========================================"
                );

                console.log(
                    "Server:",
                    "http://localhost:5000"
                );

                console.log(
                    "Face + Voice authentication: ENABLED"
                );

                console.log(
                    "Python:",
                    PYTHON_EXECUTABLE
                );

                console.log(
                    "Voice script:",
                    voiceScript
                );

                console.log(
                    "Voice threshold:",
                    VOICE_THRESHOLD
                );

                console.log(
                    "StudentDB: StudentDB.students"
                );

                console.log(
                    "LoginDB: StudentLoginDB.login_history"
                );

                console.log(
                    "Direct logout confirmation: ENABLED"
                );

                console.log(
                    "Student-specific locking: ENABLED"
                );

                console.log(
                    "========================================\n"
                );
            }
        );
    }

    catch (error) {
        console.error(
            "\n========================================"
        );

        console.error(
            "SERVER STARTUP FAILED"
        );

        console.error(
            "========================================"
        );

        console.error(
            error
        );

        console.error(
            "========================================\n"
        );
    }
}

startServer();