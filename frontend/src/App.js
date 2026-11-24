import React, { useRef, useState } from "react";
import "./App.css";

const BACKEND_URL = "http://localhost:5000";

function App() {
    const [mode, setMode] = useState("register");

    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const mediaStreamRef = useRef(null);

    const [cameraStarted, setCameraStarted] = useState(false);
    const [capturedImage, setCapturedImage] = useState(null);

    const [registrationDetails, setRegistrationDetails] = useState({
        student_id: "",
        name: "",
        department: "",
        semester: ""
    });

    const [loginDetails, setLoginDetails] = useState({
        student_id: "",
        name: "",
        department: "",
        semester: ""
    });

    const [voiceBlob, setVoiceBlob] = useState(null);
    const [isRecording, setIsRecording] = useState(false);

    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    const [message, setMessage] = useState("");
    const [error, setError] = useState("");
    const [loading, setLoading] = useState(false);

    const [recognizedStudent, setRecognizedStudent] = useState(null);

    const [authenticationStep, setAuthenticationStep] =
        useState("capture");

    const startCamera = async () => {
        try {
            setError("");
            setMessage("");

            const stream =
                await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false
                });

            mediaStreamRef.current = stream;

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }

            setCameraStarted(true);
        } catch (err) {
            console.error(err);

            setError(
                "Cannot access camera. Please allow camera permission."
            );
        }
    };

    const stopCamera = () => {
        if (mediaStreamRef.current) {
            mediaStreamRef.current
                .getTracks()
                .forEach((track) => track.stop());

            mediaStreamRef.current = null;
        }

        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }

        setCameraStarted(false);
    };

    const captureFace = () => {
        if (!videoRef.current || !canvasRef.current) {
            setError("Camera is not ready.");
            return;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const context = canvas.getContext("2d");

        context.drawImage(
            video,
            0,
            0,
            canvas.width,
            canvas.height
        );

        const image =
            canvas.toDataURL("image/jpeg");

        setCapturedImage(image);

        setMessage("Face captured successfully.");
        setError("");
    };

    const dataURLToBlob = async (dataURL) => {
        const response = await fetch(dataURL);
        return await response.blob();
    };

    const handleRegistrationChange = (e) => {
        setRegistrationDetails({
            ...registrationDetails,
            [e.target.name]: e.target.value
        });
    };

    const handleLoginChange = (e) => {
        setLoginDetails({
            ...loginDetails,
            [e.target.name]: e.target.value
        });
    };

    const startVoiceRecording = async () => {
        try {
            setError("");
            setMessage("");

            const stream =
                await navigator.mediaDevices.getUserMedia({
                    audio: true
                });

            const recorder =
                new MediaRecorder(stream);

            mediaRecorderRef.current = recorder;
            audioChunksRef.current = [];

            recorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            recorder.onstop = () => {
                const blob =
                    new Blob(
                        audioChunksRef.current,
                        {
                            type: "audio/webm"
                        }
                    );

                setVoiceBlob(blob);

                stream
                    .getTracks()
                    .forEach((track) => track.stop());

                setMessage(
                    "Voice recording completed."
                );
            };

            recorder.start();

            setIsRecording(true);

            setMessage(
                "Recording voice... Please speak clearly."
            );
        } catch (err) {
            console.error(err);

            setError(
                "Cannot access microphone. Please allow microphone permission."
            );
        }
    };

    const stopVoiceRecording = () => {
        if (
            mediaRecorderRef.current &&
            mediaRecorderRef.current.state !== "inactive"
        ) {
            mediaRecorderRef.current.stop();
        }

        setIsRecording(false);
    };

    const resetApplication = () => {
        stopCamera();

        setCapturedImage(null);
        setVoiceBlob(null);

        setMessage("");
        setError("");

        setRecognizedStudent(null);

        setAuthenticationStep("capture");

        setLoginDetails({
            student_id: "",
            name: "",
            department: "",
            semester: ""
        });
    };

    const switchMode = (newMode) => {
        resetApplication();
        setMode(newMode);
    };

    const registerStudent = async () => {
        setError("");
        setMessage("");

        const {
            student_id,
            name,
            department,
            semester
        } = registrationDetails;

        if (
            !student_id ||
            !name ||
            !department ||
            !semester
        ) {
            setError(
                "Please enter all student details."
            );

            return;
        }

        if (!capturedImage) {
            setError(
                "Please capture the student's face."
            );

            return;
        }

        if (!voiceBlob) {
            setError(
                "Please record the student's voice."
            );

            return;
        }

        try {
            setLoading(true);

            setMessage(
                "Registering student. Please wait..."
            );

            const formData = new FormData();

            formData.append(
                "student_id",
                student_id
            );

            formData.append(
                "name",
                name
            );

            formData.append(
                "department",
                department
            );

            formData.append(
                "semester",
                semester
            );

            const faceBlob =
                await dataURLToBlob(
                    capturedImage
                );

            formData.append(
                "face",
                faceBlob,
                `${student_id}_face.jpg`
            );

            formData.append(
                "voice",
                voiceBlob,
                `${student_id}_voice.webm`
            );

            const response =
                await fetch(
                    `${BACKEND_URL}/register`,
                    {
                        method: "POST",
                        body: formData
                    }
                );

            const data =
                await response.json();

            if (
                !response.ok ||
                !data.success
            ) {
                throw new Error(
                    data.message ||
                    "Registration failed."
                );
            }

            setMessage(
                `Registration successful. Student ${student_id} is now registered.`
            );

            setError("");

            setRegistrationDetails({
                student_id: "",
                name: "",
                department: "",
                semester: ""
            });

            setCapturedImage(null);
            setVoiceBlob(null);

            stopCamera();

        } catch (err) {
            console.error(err);

            setError(
                err.message ||
                "Registration failed."
            );

            setMessage("");

        } finally {
            setLoading(false);
        }
    };

    const authenticateStudent = async () => {
        setError("");
        setMessage("");

        if (!capturedImage) {
            setError(
                "Please capture the student's face."
            );

            return;
        }

        if (!voiceBlob) {
            setError(
                "Please record the student's voice."
            );

            return;
        }

        try {
            setLoading(true);

            setMessage(
                "Checking face and voice..."
            );

            const formData = new FormData();

            const faceBlob =
                await dataURLToBlob(
                    capturedImage
                );

            formData.append(
                "face",
                faceBlob,
                "login_face.jpg"
            );

            formData.append(
                "voice",
                voiceBlob,
                "login_voice.webm"
            );

            const response =
                await fetch(
                    `${BACKEND_URL}/recognize`,
                    {
                        method: "POST",
                        body: formData
                    }
                );

            const data =
                await response.json();

            if (!data.recognized) {
                setError(
                    data.message ||
                    "Student could not be recognized."
                );

                setMessage("");

                return;
            }

            if (
                data.recognized &&
                data.voiceVerified === false
            ) {
                setError(
                    data.message ||
                    "Voice verification failed."
                );

                setMessage("");

                return;
            }

            if (
                data.alreadyLoggedIn === true ||
                data.action === "ask_logout" ||
                data.action === "ask_student_details"
            ) {
                setRecognizedStudent(
                    data.student || null
                );

                setAuthenticationStep(
                    "ask_logout"
                );

                setMessage("");
                setError("");

                return;
            }

            if (
                data.action === "locked" ||
                data.locked === true
            ) {
                setRecognizedStudent(
                    data.student || null
                );

                setAuthenticationStep(
                    "locked"
                );

                setError(
                    data.message ||
                    "This student is locked for today."
                );

                setMessage("");

                return;
            }

            if (
                data.verified &&
                data.student
            ) {
                setRecognizedStudent(
                    data.student
                );

                setAuthenticationStep(
                    "capture"
                );

                setMessage(
                    "Face and voice matched. Login successful."
                );

                setError("");

                stopCamera();

                return;
            }

            setError(
                data.message ||
                "Authentication failed."
            );

        } catch (err) {
            console.error(err);

            setError(
                "Cannot connect to Backend."
            );

            setMessage("");

        } finally {
            setLoading(false);
        }
    };

    const handleYesLogout = async () => {
        if (!recognizedStudent) {
            return;
        }

        try {
            setLoading(true);
            setError("");
            setMessage(
                "Logging out..."
            );

            const response =
                await fetch(
                    `${BACKEND_URL}/logout`,
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body: JSON.stringify({
                            student_id:
                                recognizedStudent.student_id
                        })
                    }
                );

            const data =
                await response.json();

            if (
                !response.ok ||
                !data.success
            ) {
                throw new Error(
                    data.message ||
                    "Logout failed."
                );
            }

            stopCamera();

            setCapturedImage(null);
            setVoiceBlob(null);

            setAuthenticationStep(
                "locked"
            );

            setMessage("");

            setError(
                "Logout successful. This student is locked for today."
            );

        } catch (err) {
            console.error(err);

            setError(
                err.message ||
                "Logout failed."
            );

            setMessage("");

        } finally {
            setLoading(false);
        }
    };

    const handleNoLogout = () => {
        stopCamera();

        setCapturedImage(null);
        setVoiceBlob(null);

        setRecognizedStudent(null);

        setAuthenticationStep(
            "capture"
        );

        setError("");
        setMessage(
            "Logout cancelled."
        );
    };

    return (
        <div className="app">

            <header className="header">

                <h1>
                    Student Face + Voice Authentication
                </h1>

                <p>
                    Secure Student Registration and Login System
                </p>

            </header>

            <div className="mode-buttons">

                <button
                    className={
                        mode === "register"
                            ? "active"
                            : ""
                    }
                    onClick={() =>
                        switchMode("register")
                    }
                >
                    Student Registration
                </button>

                <button
                    className={
                        mode === "login"
                            ? "active"
                            : ""
                    }
                    onClick={() =>
                        switchMode("login")
                    }
                >
                    Student Authentication
                </button>

            </div>

            <main className="main-container">

                {mode === "register" && (

                    <section className="card">

                        <h2>
                            Register New Student
                        </h2>

                        <p>
                            Enter student details, capture the
                            face and record the voice.
                        </p>

                        <div className="form">

                            <input
                                type="text"
                                name="student_id"
                                placeholder="Student ID"
                                value={
                                    registrationDetails.student_id
                                }
                                onChange={
                                    handleRegistrationChange
                                }
                            />

                            <input
                                type="text"
                                name="name"
                                placeholder="Student Name"
                                value={
                                    registrationDetails.name
                                }
                                onChange={
                                    handleRegistrationChange
                                }
                            />

                            <input
                                type="text"
                                name="department"
                                placeholder="Department"
                                value={
                                    registrationDetails.department
                                }
                                onChange={
                                    handleRegistrationChange
                                }
                            />

                            <input
                                type="number"
                                name="semester"
                                placeholder="Semester"
                                value={
                                    registrationDetails.semester
                                }
                                onChange={
                                    handleRegistrationChange
                                }
                            />

                        </div>

                        <div className="camera-section">

                            <video
                                ref={videoRef}
                                autoPlay
                                playsInline
                                className="camera"
                            />

                            <canvas
                                ref={canvasRef}
                                style={{
                                    display: "none"
                                }}
                            />

                            <div className="camera-buttons">

                                {!cameraStarted ? (

                                    <button
                                        onClick={
                                            startCamera
                                        }
                                    >
                                        Start Camera
                                    </button>

                                ) : (

                                    <>
                                        <button
                                            onClick={
                                                captureFace
                                            }
                                        >
                                            Capture Face
                                        </button>

                                        <button
                                            onClick={
                                                stopCamera
                                            }
                                        >
                                            Stop Camera
                                        </button>
                                    </>

                                )}

                            </div>

                        </div>

                        {capturedImage && (

                            <div className="preview">

                                <h3>
                                    Captured Face
                                </h3>

                                <img
                                    src={
                                        capturedImage
                                    }
                                    alt="Captured face"
                                />

                            </div>

                        )}

                        <div className="voice-section">

                            <h3>
                                Voice Registration
                            </h3>

                            {!isRecording ? (

                                <button
                                    onClick={
                                        startVoiceRecording
                                    }
                                >
                                    Start Voice Recording
                                </button>

                            ) : (

                                <button
                                    onClick={
                                        stopVoiceRecording
                                    }
                                >
                                    Stop Voice Recording
                                </button>

                            )}

                            {voiceBlob && (

                                <p>
                                    ✓ Voice recording ready
                                </p>

                            )}

                        </div>

                        <button
                            className="primary-button"
                            onClick={
                                registerStudent
                            }
                            disabled={loading}
                        >
                            {loading
                                ? "Registering..."
                                : "Register Student"}
                        </button>

                    </section>

                )}

                {mode === "login" && (

                    <section className="card">

                        <h2>
                            Student Authentication
                        </h2>

                        {authenticationStep ===
                            "capture" && (

                            <>

                                <p>
                                    Capture the student's face
                                    and verify their voice.
                                </p>

                                <div className="camera-section">

                                    <video
                                        ref={videoRef}
                                        autoPlay
                                        playsInline
                                        className="camera"
                                    />

                                    <canvas
                                        ref={canvasRef}
                                        style={{
                                            display: "none"
                                        }}
                                    />

                                    <div className="camera-buttons">

                                        {!cameraStarted ? (

                                            <button
                                                onClick={
                                                    startCamera
                                                }
                                            >
                                                Start Camera
                                            </button>

                                        ) : (

                                            <>
                                                <button
                                                    onClick={
                                                        captureFace
                                                    }
                                                >
                                                    Capture Face
                                                </button>

                                                <button
                                                    onClick={
                                                        stopCamera
                                                    }
                                                >
                                                    Stop Camera
                                                </button>
                                            </>

                                        )}

                                    </div>

                                </div>

                                {capturedImage && (

                                    <div className="preview">

                                        <h3>
                                            Captured Face
                                        </h3>

                                        <img
                                            src={
                                                capturedImage
                                            }
                                            alt="Captured face"
                                        />

                                    </div>

                                )}

                                <div className="voice-section">

                                    <h3>
                                        Voice Verification
                                    </h3>

                                    {!isRecording ? (

                                        <button
                                            onClick={
                                                startVoiceRecording
                                            }
                                        >
                                            Start Voice Recording
                                        </button>

                                    ) : (

                                        <button
                                            onClick={
                                                stopVoiceRecording
                                            }
                                        >
                                            Stop Voice Recording
                                        </button>

                                    )}

                                    {voiceBlob && (

                                        <p>
                                            ✓ Voice ready
                                        </p>

                                    )}

                                </div>

                                <button
                                    className="primary-button"
                                    onClick={
                                        authenticateStudent
                                    }
                                    disabled={
                                        loading
                                    }
                                >
                                    {loading
                                        ? "Authenticating..."
                                        : "Authenticate Student"}
                                </button>

                            </>

                        )}

                        {authenticationStep ===
                            "ask_logout" && (

                            <div className="verification-box">

                                <h2>
                                    Student Already Logged In
                                </h2>

                                <p>
                                    Face and voice authentication successful.
                                </p>

                                <div className="recognized-info">

                                    <p>
                                        <strong>
                                            Student ID:
                                        </strong>{" "}
                                        {
                                            recognizedStudent
                                                ?.student_id
                                        }
                                    </p>

                                    <p>
                                        <strong>
                                            Name:
                                        </strong>{" "}
                                        {
                                            recognizedStudent
                                                ?.name
                                        }
                                    </p>

                                    <p>
                                        <strong>
                                            Department:
                                        </strong>{" "}
                                        {
                                            recognizedStudent
                                                ?.department
                                        }
                                    </p>

                                    <p>
                                        <strong>
                                            Semester:
                                        </strong>{" "}
                                        {
                                            recognizedStudent
                                                ?.semester
                                        }
                                    </p>

                                </div>

                                <h3>
                                    You are already logged in.
                                    Do you wish to logout?
                                </h3>

                                <div className="yes-no-buttons">

                                    <button
                                        onClick={
                                            handleYesLogout
                                        }
                                        disabled={
                                            loading
                                        }
                                    >
                                        {loading
                                            ? "Processing..."
                                            : "YES"}
                                    </button>

                                    <button
                                        onClick={
                                            handleNoLogout
                                        }
                                        disabled={
                                            loading
                                        }
                                    >
                                        NO
                                    </button>

                                </div>

                            </div>

                        )}

                        {authenticationStep ===
                            "locked" && (

                            <div className="locked-box">

                                <h2>
                                    Student Locked
                                </h2>

                                <p>
                                    {
                                        recognizedStudent
                                            ?.name
                                    }
                                </p>

                                <p>
                                    Student ID:{" "}
                                    {
                                        recognizedStudent
                                            ?.student_id
                                    }
                                </p>

                                <strong>
                                    This student is locked
                                    for today.
                                </strong>

                                <p>
                                    The student can authenticate
                                    again tomorrow.
                                </p>

                                <button
                                    onClick={() =>
                                        switchMode(
                                            "login"
                                        )
                                    }
                                >
                                    Back to Authentication
                                </button>

                            </div>

                        )}

                    </section>

                )}

                {message && (

                    <div className="success-message">
                        {message}
                    </div>

                )}

                {error && (

                    <div className="error-message">
                        {error}
                    </div>

                )}

            </main>

        </div>
    );
}

export default App;