
# Student Attendance Management System

A multimodal student attendance management system that combines **face recognition** and **voice verification** with a React frontend, Node.js backend, and MongoDB database.

## Features

- Student registration with face and voice-related processing.
- Face recognition using InsightFace and OpenCV.
- Speaker recognition and voice verification using SpeechBrain.
- Student recognition and attendance-related login/logout operations.
- REST API built with Express.
- MongoDB integration for student records and login history.
- File upload handling through Multer.

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React 19 |
| Backend | Node.js, Express 5 |
| Database | MongoDB |
| Face Recognition | Python, OpenCV, InsightFace |
| Voice Recognition | Python, SpeechBrain, PyTorch, PyAudio |
| Database Integration | MongoDB Node.js Driver, PyMongo |
| File Uploads | Multer |

## Project Structure

```text
student_attend/
├── Backend/
│   ├── server.js
│   └── package.json
├── frontend/
│   ├── public/
│   ├── src/
│   └── package.json
├── face_engine/
│   ├── recognition_engine.py
│   ├── recognize.py
│   ├── register.py
│   ├── speaker_test.py
│   └── voice_embedding.py
├── mongoDB/
├── voice_engine/
├── .gitignore
└── README.md
```

## Backend API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Backend status |
| POST | `/register` | Student registration |
| POST | `/recognize` | Student recognition |
| POST | `/logout` | Logout operation |
| POST | `/voice/verify` | Voice verification |

## Database

The application connects to a local MongoDB instance.

- `StudentDB.students` — student records.
- `StudentLoginDB.login_history` — login history.

Default MongoDB connection:

`mongodb://localhost:27017/`

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd student_attend
```

### 2. Start MongoDB

Make sure MongoDB is installed and running locally.

### 3. Start the backend

```bash
cd Backend
npm install
npm start
```

### 4. Start the frontend

Open another terminal:

```bash
cd frontend
npm install
npm start
```

### 5. Python environment

Install the required libraries:

```bash
pip install opencv-python numpy insightface pymongo torch pyaudio speechbrain
```

Additional system dependencies or model downloads may be required.

## Security Notes

- Do not commit `.env` files, credentials, database secrets, or personal student data.
- Keep temporary uploads and generated files out of Git.
- Configure database credentials and deployment settings securely before production use.

## Project Status

This repository contains the React frontend, Express backend, and Python face and voice recognition modules. Deployment configuration and production readiness should be validated separately.

---

**Student Attendance Management System**

Built with React, Node.js, MongoDB, and Python.
