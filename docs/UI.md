Master Design & Implementation Prompt: Smart OMR Grader

1. Project Context & Vision

Project Title: Smart OMR Grader (Mobile App)
Objective: A professional academic tool that uses a C++ image processing core (via JNI/NDK) to grade multiple-choice OMR sheets in real-time using a smartphone camera.
Target Users: Educators and Exam Administrators.
Core Tech Stack: Android (Kotlin/Java), Android NDK (C++), OpenCV, CameraX, SQLite/Room, Google Drive API.

2. Design Language (Figma Guidelines)

Aesthetic: Clean, Academic, and Trustworthy.

Color Palette:

Primary: #2563EB (Trust Blue) - Headers, Primary Buttons.

Secondary: #0F9D58 (Success Green) - Grading results, Capture buttons.

Background: #F9FAFB (Cool Gray) - App surfaces.

Typography: Inter or System Sans-Serif. Bold weights for student IDs and scores.

Key Components: * Elevated CardViews (12dp radius) for exam lists.

Modern Bottom Navigation for Dashboard, Exams, and Settings.

Floating Action Button (FAB) for "Quick Scan."

3. UI Screen Requirements (Matching omr_app_layouts.xml)

Screen A: Teacher's Dashboard

Header: Dynamic greeting with summary of total students and active classes.

Class Cards: Display Class Name, Student Count, and a progress bar of recent grading activity.

Recent Activity: A list showing the last 3-5 exams scanned with "Sync Status" to Google Drive.

Screen B: Exam Management & Results

Analytics Overview: 3-column grid showing "Total Scanned," "Average Score," and "Highest/Lowest" marks.

Scanned List: Scrollable list of students. Each row must show: Student ID, Raw Score (e.g., 38/40), and a "Verified" vs "Needs Review" badge.

Actions: Floating "Start Scoring" button and an "Export" action in the top app bar.

Screen C: Real-time Camera Interface (The "Scanning" Mode)

AR Overlay: A semi-transparent overlay with a precise bounding box matching the 4-anchor system defined in the ProfessionalGenerator script.

Visual Feedback: * State 1 (Searching): Red boundary/corners.

State 2 (Locked): Green boundary with a "Ready" indicator.

State 3 (Processed): Temporary score toast appearing at the bottom-right for 1.5 seconds.

Controls: Toggle for Flash (Auto/On/Off) and a Manual Capture button for difficult lighting.

Screen D: Export & Integration

Format Selection: Radio buttons for .XLSX, .CSV, or .PDF.

Column Mapping: Toggle switches for "Include Student ID," "Include Question Details," and "Include Timestamp."

Cloud Status: Integration status indicator for Google Drive (Account name & last sync time).

4. Interaction & Logic Constraints

Latency: The JNI call must return results in <500ms to maintain a fluid camera preview (FPS > 1).

Concurrency: Use background threads for JNI image processing; the UI must remain responsive during the grading "Capture" event.

Error Handling: Implement custom modals (no system alerts) for "Low Light Detected," "Anchor Misalignment," or "No Student ID Found."

5. Development Timeline (Aligned with Project Plan)

Current Milestone: Checkpoint 4 - Integrating CameraX and Real-time JNI data.

Upcoming Milestone: Checkpoint 5 - Finalizing Data Persistence (Room) and Google Drive Export logic.