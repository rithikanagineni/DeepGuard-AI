import os
import tempfile
from datetime import datetime
from html import escape

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.predictor import predict_image, predict_video


# ---------------------------------------------------------
# APP CONFIGURATION
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(
    title="AI-Based Deepfake Detection System",
    description="CNN-SVM based deepfake detection for images and videos.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home():

    index_path = os.path.join(
        STATIC_DIR,
        "index.html"
    )

    with open(
        index_path,
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
async def health_check():

    return {
        "status": "healthy",
        "message": "AI-Based Deepfake Detection System API is running."
    }


# ---------------------------------------------------------
# IMAGE PREDICTION
# ---------------------------------------------------------

@app.post("/predict/image")
async def predict_image_endpoint(
    file: UploadFile = File(...)
):

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    filename = (
        file.filename
        or "uploaded_image"
    )

    extension = os.path.splitext(
        filename
    )[1].lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Unsupported image format."
        )

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty."
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(contents)
            temp_path = temp_file.name

        result = predict_image(
            temp_path
        )

        result["filename"] = filename
        result["media_type"] = "image"

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Image prediction failed: {str(e)}"
        )

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            os.remove(temp_path)


# ---------------------------------------------------------
# VIDEO PREDICTION
# ---------------------------------------------------------

@app.post("/predict/video")
async def predict_video_endpoint(
    file: UploadFile = File(...)
):

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    }

    filename = (
        file.filename
        or "uploaded_video"
    )

    extension = os.path.splitext(
        filename
    )[1].lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
        )

    contents = await file.read()

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="The uploaded video is empty."
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(contents)
            temp_path = temp_file.name

        result = predict_video(
            temp_path
        )

        result["filename"] = filename
        result["media_type"] = "video"

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Video prediction failed: {str(e)}"
        )

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            os.remove(temp_path)


# ---------------------------------------------------------
# DETECTION REPORT
# ---------------------------------------------------------

@app.post("/report")
async def generate_report(result: dict):

    if not result:

        raise HTTPException(
            status_code=400,
            detail="No detection result was provided."
        )


    # -----------------------------------------------------
    # EXTRACT RESULT DATA
    # -----------------------------------------------------

    filename = str(
        result.get(
            "filename",
            "Unknown media"
        )
    )

    media_type = str(
        result.get(
            "media_type",
            "Unknown"
        )
    )

    label = str(
        result.get(
            "label",
            "Not available"
        )
    )

    model_score = result.get(
        "model_score",
        None
    )

    confidence = result.get(
        "confidence",
        None
    )

    analyzed_frames = result.get(
        "analyzed_frames",
        None
    )

    face_detected_frames = result.get(
        "face_detected_frames",
        None
    )

    face_detection_percentage = result.get(
        "face_detection_percentage",
        None
    )

    real_frames = result.get(
        "real_frames",
        None
    )

    fake_frames = result.get(
        "fake_frames",
        None
    )

    suspicious_frames = result.get(
        "suspicious_frames",
        None
    )

    fake_frame_percentage = result.get(
        "fake_frame_percentage",
        None
    )

    median_score = result.get(
        "median_score",
        None
    )

    status = str(
        result.get(
            "status",
            "Unknown"
        )
    )

    file_size = result.get("file_size")
    file_type = result.get("file_type")
    image_width = result.get("image_width")
    image_height = result.get("image_height")
    video_width = result.get("video_width")
    video_height = result.get("video_height")
    video_duration = result.get("video_duration")
    video_fps = result.get("video_fps")
    frame_results = result.get("frame_results")
    if not isinstance(frame_results, list):
        frame_results = []


    # -----------------------------------------------------
    # SAFE DISPLAY VALUES
    # -----------------------------------------------------

    filename_display = escape(
        filename
    )

    media_type_display = escape(
        media_type.upper()
    )

    label_display = escape(
        label
    )

    status_display = escape(
        status
    )


    # -----------------------------------------------------
    # FORMAT NUMERIC VALUES
    # -----------------------------------------------------

    if isinstance(
        model_score,
        (int, float)
    ):

        score_display = (
            f"{model_score:.2f}%"
        )

    else:

        score_display = "N/A"


    if isinstance(
        confidence,
        (int, float)
    ):

        confidence_display = (
            f"{confidence:.2f}%"
        )

    else:

        confidence_display = "N/A"


    file_size_display = (
        f"{file_size:,} bytes"
        if isinstance(file_size, int) and file_size >= 0
        else "N/A"
    )

    width = image_width or video_width
    height = image_height or video_height
    resolution_display = (
        escape(f"{width} × {height} px")
        if isinstance(width, int) and isinstance(height, int)
        else "N/A"
    )

    duration_display = (
        f"{float(video_duration):.2f} seconds"
        if isinstance(video_duration, (int, float))
        else "N/A"
    )

    fps_display = (
        f"{float(video_fps):.2f} FPS"
        if isinstance(video_fps, (int, float))
        else "N/A"
    )

    file_type_display = escape(
        str(file_type or "Not available")
    )

    fingerprint_display = escape(
        str(result.get("file_fingerprint") or "Not available")
    )

    frame_table_rows = []
    suspicious_timestamps = []
    for frame in frame_results[:20]:
        if not isinstance(frame, dict):
            continue

        frame_number = frame.get("frame_number")
        timestamp = frame.get("timestamp")
        assessment = str(frame.get("assessment") or "Not analyzed")
        face_status = (
            "Detected"
            if frame.get("face_detected") is True
            else "Not detected"
            if frame.get("face_detected") is False
            else "Not available"
        )
        frame_score = frame.get("model_score")
        frame_score_display = (
            f"{float(frame_score):.2f}%"
            if isinstance(frame_score, (int, float))
            else "Not available"
        )
        timestamp_display = (
            f"{float(timestamp):.2f}s"
            if isinstance(timestamp, (int, float))
            else "Not available"
        )
        frame_table_rows.append(
            "<tr>"
            f"<td>{escape(str(frame_number or 'N/A'))}</td>"
            f"<td>{escape(timestamp_display)}</td>"
            f"<td>{escape(face_status)}</td>"
            f"<td>{escape(frame_score_display)}</td>"
            f"<td>{escape(assessment)}</td>"
            "</tr>"
        )
        if assessment in {"Fake", "Needs Review"}:
            suspicious_timestamps.append(
                f"Frame {frame_number or 'N/A'} at {timestamp_display}"
            )

    frame_table_display = (
        "".join(frame_table_rows)
        or '<tr><td colspan="5">No sampled frame results available.</td></tr>'
    )
    suspicious_timestamps_display = escape(
        ", ".join(suspicious_timestamps)
        if suspicious_timestamps
        else "No sampled frames fell within the Needs Review band."
    )

    face_evidence_display = (
        f"{face_detected_frames}/{analyzed_frames} sampled frames "
        f"({face_detection_percentage:.1f}%)"
        if isinstance(face_detected_frames, int)
        and isinstance(analyzed_frames, int)
        and analyzed_frames > 0
        and isinstance(face_detection_percentage, (int, float))
        else (
            "Face detected"
            if result.get("face_detected") is True
            else "No face detected"
            if result.get("face_detected") is False
            else "Not available"
        )
    )

    evidence_score_status = (
        "CNN facial-feature extraction and SVM classification produced a score."
        if isinstance(model_score, (int, float))
        else "CNN/SVM score unavailable; no usable facial features were classified."
    )

    agreement_display = (
        f"{float(result['model_agreement_percentage']):.1f}% sampled-frame agreement"
        if isinstance(result.get("model_agreement_percentage"), (int, float))
        else "Not available"
    )

    def safe_count(value):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return "N/A"

    analyzed_frames_display = safe_count(analyzed_frames)
    face_detected_frames_display = safe_count(face_detected_frames)
    real_frames_display = safe_count(real_frames)
    fake_frames_display = safe_count(fake_frames)
    suspicious_frames_display = safe_count(suspicious_frames)


    if isinstance(
        median_score,
        (int, float)
    ):

        median_score_display = (
            f"{median_score:.4f}"
        )

    else:

        median_score_display = "N/A"


    if isinstance(
        face_detection_percentage,
        (int, float)
    ):

        face_detection_display = (
            f"{face_detection_percentage:.2f}%"
        )

    else:

        face_detection_display = "N/A"


    if isinstance(
        fake_frame_percentage,
        (int, float)
    ):

        fake_frame_display = (
            f"{fake_frame_percentage:.2f}%"
        )

    else:

        fake_frame_display = "N/A"


    # -----------------------------------------------------
    # RESULT STYLING
    # -----------------------------------------------------

    if label.lower() == "fake":

        result_title = (
            "Potentially Manipulated Media"
        )

        result_description = (
            "The model detected visual patterns "
            "associated with manipulated media."
        )

        result_class = "fake"

        result_icon = "!"

    elif label.lower() == "real":

        result_title = (
            "Media Classified as Real"
        )

        result_description = (
            "The model classified the submitted "
            "media as Real based on the learned "
            "visual features."
        )

        result_class = "real"

        result_icon = "✓"

    elif label.lower() == "needs review":

        result_title = (
            "Needs Review"
        )

        result_description = (
            "The available evidence is inconclusive. "
            "Manual review is recommended."
        )

        result_class = "review"

        result_icon = "?"

    else:

        result_title = (
            "Detection Inconclusive"
        )

        result_description = (
            "The system could not produce a valid "
            "final classification."
        )

        result_class = "unknown"

        result_icon = "?"


    # -----------------------------------------------------
    # FRAME DISTRIBUTION
    # -----------------------------------------------------

    fake_width = (
        max(0, min(100, float(fake_frame_percentage)))
        if isinstance(fake_frame_percentage, (int, float))
        else 0
    )
    suspicious_frame_percentage = result.get(
        "suspicious_frame_percentage",
        0
    )
    suspicious_width = (
        max(0, min(100 - fake_width, float(suspicious_frame_percentage)))
        if isinstance(suspicious_frame_percentage, (int, float))
        else 0
    )
    real_width = max(
        0,
        100 - fake_width - suspicious_width
    )

    if (
        isinstance(
            fake_frame_percentage,
            (int, float)
        )
    ):

        fake_width = max(0, min(100, fake_frame_percentage))
        suspicious_width = max(
            0,
            min(100 - fake_width, suspicious_width)
        )
        real_width = max(0, 100 - fake_width - suspicious_width)


    # -----------------------------------------------------
    # VIDEO SECTION
    # -----------------------------------------------------

    video_section = ""

    if media_type.lower() == "video":

        video_section = f"""

        <section class="section">

            <div class="section-heading">

                <div class="section-number">
                    03
                </div>

                <div>
                    <h2>Video Analysis</h2>
                    <p>
                        Frame-level analysis performed by DeepGuard AI
                    </p>
                </div>

            </div>


            <div class="stats-grid">

                <div class="stat-card">

                    <div class="stat-icon">
                        ◫
                    </div>

                    <span>
                        Frames Analyzed
                    </span>

                    <strong>
                        {
                            analyzed_frames_display
                        }
                    </strong>

                </div>

                <div class="stat-card">
                    <div class="stat-icon">?</div>
                    <span>Needs Review Frames</span>
                    <strong>{suspicious_frames_display}</strong>
                </div>


                <div class="stat-card">

                    <div class="stat-icon">
                        ◉
                    </div>

                    <span>
                        Face Frames
                    </span>

                    <strong>
                        {
                            face_detected_frames_display
                        }
                    </strong>

                </div>


                <div class="stat-card">

                    <div class="stat-icon">
                        ✓
                    </div>

                    <span>
                        Face Detection
                    </span>

                    <strong>
                        {face_detection_display}
                    </strong>

                </div>


                <div class="stat-card">

                    <div class="stat-icon">
                        R
                    </div>

                    <span>
                        Real Frames
                    </span>

                    <strong>
                        {
                            real_frames_display
                        }
                    </strong>

                </div>


                <div class="stat-card">

                    <div class="stat-icon">
                        F
                    </div>

                    <span>
                        Fake Frames
                    </span>

                    <strong>
                        {
                            fake_frames_display
                        }
                    </strong>

                </div>


                <div class="stat-card">

                    <div class="stat-icon">
                        %
                    </div>

                    <span>
                        Fake Frame Ratio
                    </span>

                    <strong>
                        {fake_frame_display}
                    </strong>

                </div>

            </div>


            <div class="distribution-card">

                <div class="distribution-header">

                    <div>
                        <span>
                            FRAME DISTRIBUTION
                        </span>

                        <h3>
                            Sampled Frame Assessments
                        </h3>
                    </div>

                    <strong>
                        {fake_frame_display} Fake
                    </strong>

                </div>


                <div class="distribution-bar">

                    <div
                        class="real-segment"
                        style="width:{real_width}%"
                    ></div>

                    <div
                        class="review-segment"
                        style="width:{suspicious_width}%"
                    ></div>

                    <div
                        class="fake-segment"
                        style="width:{fake_width}%"
                    ></div>

                </div>


                <div class="distribution-legend">

                    <span>
                        <i class="dot real-dot"></i>
                        Real Frames
                    </span>

                    <span>
                        <i class="dot fake-dot"></i>
                        Potentially Fake Frames
                    </span>

                    <span>
                        <i class="dot review-dot"></i>
                        Needs Review
                    </span>

                </div>

            </div>


            <div class="median-card">

                <div>

                    <span>
                        MEDIAN MODEL SCORE
                    </span>

                    <strong>
                        {median_score_display}
                    </strong>

                </div>

                <p>
                    Median SVM decision score across
                    analyzed video frames.
                </p>

            </div>

        </section>

        """


    # -----------------------------------------------------
    # GENERATED DATE
    # -----------------------------------------------------

    generated_at = datetime.now().strftime(
        "%d %B %Y, %I:%M:%S %p"
    )


    # -----------------------------------------------------
    # REPORT HTML
    # -----------------------------------------------------

    report_html = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
    DeepGuard AI Forensic Analysis Report
</title>


<style>

/* =====================================================
   RESET
   ===================================================== */

* {{
    box-sizing: border-box;
}}


/* =====================================================
   BODY
   ===================================================== */

body {{

    margin: 0;

    padding: 40px 20px;

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;

    background:
        linear-gradient(
            135deg,
            #f5f7ff 0%,
            #f8f9fc 50%,
            #f2f4fb 100%
        );

    color: #182033;

}}


/* =====================================================
   MAIN REPORT
   ===================================================== */

.report {{

    width: 100%;

    max-width: 980px;

    margin: auto;

    background: #ffffff;

    border-radius: 24px;

    overflow: hidden;

    box-shadow:
        0 25px 70px
        rgba(42, 48, 85, 0.12);

}}


/* =====================================================
   HEADER
   ===================================================== */

.header {{

    position: relative;

    padding: 42px 45px;

    background:
        linear-gradient(
            135deg,
            #4338ca 0%,
            #5b5ce2 45%,
            #7c3aed 100%
        );

    color: white;

    overflow: hidden;

}}


.header::before {{

    content: "";

    position: absolute;

    width: 260px;

    height: 260px;

    border-radius: 50%;

    background:
        rgba(255,255,255,0.08);

    right: -90px;

    top: -110px;

}}


.header::after {{

    content: "";

    position: absolute;

    width: 180px;

    height: 180px;

    border-radius: 50%;

    background:
        rgba(255,255,255,0.05);

    right: 120px;

    bottom: -120px;

}}


.brand {{

    position: relative;

    z-index: 2;

    display: flex;

    align-items: center;

    gap: 15px;

}}


.brand-icon {{

    width: 48px;

    height: 48px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 14px;

    background:
        rgba(255,255,255,0.15);

    border:
        1px solid
        rgba(255,255,255,0.22);

    font-size: 23px;

    font-weight: 700;

}}


.brand-name {{

    font-size: 14px;

    font-weight: 700;

    letter-spacing: 1.8px;

    text-transform: uppercase;

    opacity: 0.9;

}}


.header h1 {{

    position: relative;

    z-index: 2;

    margin: 28px 0 8px;

    font-size: 34px;

    line-height: 1.15;

    letter-spacing: -0.8px;

}}


.header p {{

    position: relative;

    z-index: 2;

    margin: 0;

    font-size: 15px;

    opacity: 0.88;

}}


.report-meta {{

    position: relative;

    z-index: 2;

    display: flex;

    flex-wrap: wrap;

    gap: 10px;

    margin-top: 25px;

}}


.meta-pill {{

    padding: 8px 12px;

    border-radius: 999px;

    background:
        rgba(255,255,255,0.12);

    border:
        1px solid
        rgba(255,255,255,0.16);

    font-size: 11px;

    font-weight: 600;

}}


/* =====================================================
   CONTENT
   ===================================================== */

.content {{

    padding: 42px 45px;

}}


/* =====================================================
   SECTION
   ===================================================== */

.section {{

    margin-bottom: 38px;

}}


.section-heading {{

    display: flex;

    align-items: center;

    gap: 14px;

    margin-bottom: 20px;

}}


.section-number {{

    width: 34px;

    height: 34px;

    flex-shrink: 0;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 10px;

    background: #eef2ff;

    color: #4f46e5;

    font-size: 12px;

    font-weight: 800;

}}


.section-heading h2 {{

    margin: 0;

    color: #20263a;

    font-size: 20px;

}}


.section-heading p {{

    margin: 4px 0 0;

    color: #8a91a3;

    font-size: 12px;

}}


/* =====================================================
   RESULT CARD
   ===================================================== */

.result-card {{

    position: relative;

    padding: 30px;

    border-radius: 18px;

    overflow: hidden;

}}


.result-card.real {{

    background:
        linear-gradient(
            135deg,
            #ecfdf5,
            #f0fdf9
        );

    border:
        1px solid #bcebd8;

}}


.result-card.fake {{

    background:
        linear-gradient(
            135deg,
            #fff1f2,
            #fff7f7
        );

    border:
        1px solid #fecdd3;

}}

.result-card.review {{
    background: #fff8eb;
    border: 1px solid #f0d7a8;
}}


.result-card.unknown {{

    background:
        linear-gradient(
            135deg,
            #f8fafc,
            #f4f6fa
        );

    border:
        1px solid #e2e8f0;

}}


.result-top {{

    display: flex;

    align-items: center;

    gap: 18px;

}}


.result-icon {{

    width: 58px;

    height: 58px;

    flex-shrink: 0;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 17px;

    font-size: 26px;

    font-weight: 800;

}}


.real .result-icon {{

    background: #d7f8e9;

    color: #079669;

}}


.fake .result-icon {{

    background: #ffe0e4;

    color: #d93650;

}}


.unknown .result-icon {{

    background: #e9edf3;

    color: #64748b;

}}


.result-title {{

    margin: 0 0 6px;

    font-size: 21px;

    color: #20263a;

}}


.result-label {{

    font-size: 12px;

    font-weight: 800;

    letter-spacing: 1px;

    text-transform: uppercase;

    color: #737b8f;

}}


.result-description {{

    margin: 22px 0 0;

    color: #596174;

    line-height: 1.65;

    font-size: 14px;

}}


/* =====================================================
   INFO GRID
   ===================================================== */

.info-grid {{

    display: grid;

    grid-template-columns:
        repeat(2, minmax(0, 1fr));

    gap: 14px;

}}


.info-card {{

    padding: 19px;

    border-radius: 14px;

    background: #f8fafc;

    border:
        1px solid #e8ebf1;

}}


.info-card span {{

    display: block;

    margin-bottom: 7px;

    color: #8991a2;

    font-size: 11px;

    font-weight: 700;

    letter-spacing: 0.6px;

    text-transform: uppercase;

}}


.info-card strong {{

    display: block;

    color: #242b3d;

    font-size: 14px;

    line-height: 1.45;

    word-break: break-word;

}}


/* =====================================================
   MODEL SCORE
   ===================================================== */

.score-grid {{

    display: grid;

    grid-template-columns:
        repeat(2, minmax(0, 1fr));

    gap: 16px;

}}


.score-card {{

    padding: 24px;

    border-radius: 16px;

    background:
        linear-gradient(
            135deg,
            #f8f7ff,
            #f5f3ff
        );

    border:
        1px solid #e5e1ff;

}}


.score-card span {{

    display: block;

    color: #7d8193;

    font-size: 11px;

    font-weight: 700;

    letter-spacing: 0.7px;

    text-transform: uppercase;

    margin-bottom: 9px;

}}


.score-card strong {{

    display: block;

    color: #4f46e5;

    font-size: 27px;

}}


/* =====================================================
   VIDEO STATS
   ===================================================== */

.stats-grid {{

    display: grid;

    grid-template-columns:
        repeat(3, minmax(0, 1fr));

    gap: 14px;

}}


.stat-card {{

    padding: 19px;

    border-radius: 14px;

    background: #f8fafc;

    border:
        1px solid #e8ebf1;

}}


.stat-icon {{

    width: 30px;

    height: 30px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 9px;

    background: #eef2ff;

    color: #5b5ce2;

    font-size: 12px;

    font-weight: 800;

    margin-bottom: 13px;

}}


.stat-card span {{

    display: block;

    color: #8991a2;

    font-size: 11px;

    font-weight: 700;

    text-transform: uppercase;

    letter-spacing: 0.5px;

    margin-bottom: 7px;

}}


.stat-card strong {{

    color: #242b3d;

    font-size: 20px;

}}


/* =====================================================
   DISTRIBUTION
   ===================================================== */

.distribution-card {{

    margin-top: 18px;

    padding: 24px;

    border-radius: 16px;

    background: #ffffff;

    border:
        1px solid #e8ebf1;

}}


.distribution-header {{

    display: flex;

    justify-content: space-between;

    align-items: flex-start;

    gap: 20px;

    margin-bottom: 20px;

}}


.distribution-header span {{

    color: #9299a9;

    font-size: 10px;

    font-weight: 800;

    letter-spacing: 1px;

}}


.distribution-header h3 {{

    margin: 5px 0 0;

    font-size: 15px;

    color: #252b3d;

}}


.distribution-header strong {{

    color: #4f46e5;

    font-size: 14px;

    white-space: nowrap;

}}


.distribution-bar {{

    width: 100%;

    height: 14px;

    display: flex;

    overflow: hidden;

    border-radius: 999px;

    background: #edf0f5;

}}


.real-segment {{

    height: 100%;

    background: #22c997;

}}


.fake-segment {{

    height: 100%;

    background: #ef6679;

}}

.review-segment {{
    height: 100%;
    background: #e1a33a;
}}


.distribution-legend {{

    display: flex;

    gap: 25px;

    margin-top: 14px;

    color: #737b8d;

    font-size: 11px;

}}


.distribution-legend span {{

    display: flex;

    align-items: center;

    gap: 7px;

}}


.dot {{

    width: 8px;

    height: 8px;

    border-radius: 50%;

    display: inline-block;

}}


.real-dot {{

    background: #22c997;

}}


.fake-dot {{

    background: #ef6679;

}}

.review-dot {{
    background: #e1a33a;
}}


/* =====================================================
   MEDIAN
   ===================================================== */

.median-card {{

    display: flex;

    justify-content: space-between;

    align-items: center;

    gap: 25px;

    margin-top: 14px;

    padding: 20px 22px;

    border-radius: 14px;

    background: #f8fafc;

    border:
        1px solid #e8ebf1;

}}


.median-card span {{

    display: block;

    margin-bottom: 6px;

    color: #8991a2;

    font-size: 10px;

    font-weight: 800;

    letter-spacing: 0.7px;

}}


.median-card strong {{

    color: #4f46e5;

    font-size: 21px;

}}


.median-card p {{

    margin: 0;

    color: #7b8394;

    font-size: 11px;

    line-height: 1.5;

    text-align: right;

}}


/* =====================================================
   NOTE
   ===================================================== */

.note {{

    padding: 21px 22px;

    border-radius: 14px;

    background: #f8fafc;

    border-left:
        4px solid #6366f1;

    color: #626a7c;

    font-size: 13px;

    line-height: 1.75;

}}

.report-table-wrap {{
    overflow-x: auto;
}}


/* =====================================================
   FOOTER
   ===================================================== */

.footer {{

    padding: 26px 45px;

    background: #f8fafc;

    border-top:
        1px solid #e8ebf1;

    color: #8a91a1;

    font-size: 11px;

    line-height: 1.7;

}}


.footer strong {{

    color: #4f46e5;

    font-size: 12px;

}}


/* =====================================================
   PRINT
   ===================================================== */

@media print {{

    body {{

        padding: 0;

        background: white;

    }}


    .report {{

        max-width: none;

        border-radius: 0;

        box-shadow: none;

    }}

}}


/* =====================================================
   MOBILE
   ===================================================== */

@media (max-width: 700px) {{

    body {{

        padding: 10px;

    }}


    .header {{

        padding: 30px 25px;

    }}


    .header h1 {{

        font-size: 27px;

    }}


    .content {{

        padding: 30px 25px;

    }}


    .info-grid,
    .score-grid,
    .stats-grid {{

        grid-template-columns: 1fr;

    }}


    .distribution-header {{

        flex-direction: column;

    }}


    .median-card {{

        align-items: flex-start;

        flex-direction: column;

    }}


    .median-card p {{

        text-align: left;

    }}


    .footer {{

        padding: 25px;

    }}

}}

</style>

</head>


<body>


<div class="report">


    <!-- =================================================
         HEADER
         ================================================= -->

    <header class="header">

        <div class="brand">

            <div class="brand-icon">
                ◈
            </div>

            <div class="brand-name">
                DeepGuard AI
            </div>

        </div>


        <h1>
            DeepGuard AI Forensic Analysis Report
        </h1>


        <p>
            CNN Feature Extraction + SVM Classification
        </p>


        <div class="report-meta">

            <span class="meta-pill">
                Automated Analysis
            </span>

            <span class="meta-pill">
                {media_type_display}
            </span>

            <span class="meta-pill">
                {generated_at}
            </span>

        </div>

    </header>


    <!-- =================================================
         CONTENT
         ================================================= -->

    <main class="content">


        <!-- DETECTION SUMMARY -->

        <section class="section">

            <div class="section-heading">

                <div class="section-number">
                    01
                </div>

                <div>

                    <h2>
                        Detection Summary
                    </h2>

                    <p>
                        Primary classification generated by the model
                    </p>

                </div>

            </div>


            <div class="result-card {result_class}">

                <div class="result-top">

                    <div class="result-icon">
                        {result_icon}
                    </div>

                    <div>

                        <h3 class="result-title">
                            {result_title}
                        </h3>

                        <div class="result-label">
                            Classification: {label_display}
                        </div>

                    </div>

                </div>


                <p class="result-description">
                    {result_description}
                </p>

            </div>

        </section>


        <!-- MEDIA INFORMATION -->

        <section class="section">

            <div class="section-heading">

                <div class="section-number">
                    02
                </div>

                <div>

                    <h2>
                        Media Information
                    </h2>

                    <p>
                        Details associated with the analyzed submission
                    </p>

                </div>

            </div>


            <div class="info-grid">

                <div class="info-card">

                    <span>
                        Filename
                    </span>

                    <strong>
                        {filename_display}
                    </strong>

                </div>


                <div class="info-card">

                    <span>
                        Media Type
                    </span>

                    <strong>
                        {media_type_display}
                    </strong>

                </div>


                <div class="info-card">

                    <span>
                        Detection Status
                    </span>

                    <strong>
                        {status_display}
                    </strong>

                </div>


                <div class="info-card">

                    <span>
                        Generated On
                    </span>

                    <strong>
                        {generated_at}
                    </strong>

                </div>

                <div class="info-card">
                    <span>File Type</span>
                    <strong>{file_type_display}</strong>
                </div>

                <div class="info-card">
                    <span>File Size</span>
                    <strong>{file_size_display}</strong>
                </div>

                <div class="info-card">
                    <span>Resolution</span>
                    <strong>{resolution_display}</strong>
                </div>

                <div class="info-card">
                    <span>Duration</span>
                    <strong>{duration_display}</strong>
                </div>

                <div class="info-card">
                    <span>Frame Rate</span>
                    <strong>{fps_display}</strong>
                </div>

                <div class="info-card">
                    <span>SHA-256 Fingerprint</span>
                    <strong>{fingerprint_display}</strong>
                </div>

            </div>

        </section>


        <!-- MODEL ANALYSIS -->

        <section class="section">

            <div class="section-heading">

                <div class="section-number">
                    03
                </div>

                <div>

                    <h2>
                        Model Analysis
                    </h2>

                    <p>
                        Output generated by the CNN feature extraction and SVM classifier
                    </p>

                </div>

            </div>


            <div class="score-grid">

                <div class="score-card">

                    <span>
                        Fake-class Model Score
                    </span>

                    <strong>
                        {score_display}
                    </strong>

                </div>


                <div class="score-card">

                    <span>
                        Class-aligned score (not calibrated)
                    </span>

                    <strong>
                        {confidence_display}
                    </strong>

                </div>

            </div>

        </section>


        <!-- VIDEO ANALYSIS -->

        {video_section}

        <section class="section">
            <div class="section-heading">
                <div class="section-number">04</div>
                <div>
                    <h2>Explainable Evidence &amp; Analysis Summary</h2>
                    <p>Only signals actually produced by this analysis are reported</p>
                </div>
            </div>

            <div class="info-grid">
                <div class="info-card"><span>Face detection</span><strong>{escape(face_evidence_display)}</strong></div>
                <div class="info-card"><span>CNN / SVM classification</span><strong>{escape(evidence_score_status)}</strong></div>
                <div class="info-card"><span>Sampled-frame agreement</span><strong>{escape(agreement_display)}</strong></div>
                <div class="info-card"><span>Temporal motion consistency</span><strong>Not analyzed</strong></div>
                <div class="info-card"><span>Frames analyzed</span><strong>{analyzed_frames_display}</strong></div>
                <div class="info-card"><span>Real-like frames</span><strong>{real_frames_display}</strong></div>
                <div class="info-card"><span>Fake-like frames</span><strong>{fake_frames_display}</strong></div>
                <div class="info-card"><span>Suspicious frames</span><strong>{suspicious_frames_display}</strong></div>
                <div class="info-card"><span>Evidence strength</span><strong>{escape(str(result.get("evidence_strength") or "Not available"))}</strong></div>
                <div class="info-card"><span>Texture / blending / lighting / provenance</span><strong>Not analyzed</strong></div>
            </div>
            <p class="note">Evidence strength is a rule-based coverage summary, not a calibrated measure of truth or model confidence.</p>
        </section>

        <section class="section">
            <div class="section-heading">
                <div class="section-number">05</div>
                <div>
                    <h2>Frame Investigation</h2>
                    <p>Sampled frames only; at most 20 frames are processed</p>
                </div>
            </div>

            <p class="note"><strong>Suspicious frame timestamps:</strong> {suspicious_timestamps_display}</p>
            <div class="report-table-wrap">
                <table class="confusion-matrix">
                    <thead>
                        <tr><th>Frame</th><th>Timestamp</th><th>Face</th><th>Model score</th><th>Assessment</th></tr>
                    </thead>
                    <tbody>{frame_table_display}</tbody>
                </table>
            </div>
        </section>

        <section class="section">
            <div class="section-heading">
                <div class="section-number">06</div>
                <div>
                    <h2>Model Information &amp; Limitations</h2>
                    <p>Project implementation and interpretation limits</p>
                </div>
            </div>
            <div class="note">
                <strong>CNN feature extractor:</strong> MobileNetV2 with ImageNet weights and pooled features.<br>
                <strong>Classifier:</strong> SVM; the displayed bounded score is derived from its decision function, not a calibrated probability.<br>
                <strong>Dataset used during development:</strong> FaceForensics++ subset documented by this project.<br><br>
                DeepGuard AI provides an AI-assisted assessment and should not be treated as absolute proof of authenticity or manipulation. Results may be affected by video quality, compression, lighting, face visibility, dataset characteristics, and unseen manipulation techniques.
            </div>
            <p class="note"><strong>Metadata note:</strong> Metadata can provide useful context but does not by itself prove whether media is authentic. Embedded metadata and provenance are not analyzed by this application.</p>
        </section>


        <!-- METHODOLOGY -->

        <section class="section">

            <div class="section-heading">

                <div class="section-number">
                    07
                </div>

                <div>

                    <h2>
                        Methodology
                    </h2>

                    <p>
                        Processing pipeline used for the detection
                    </p>

                </div>

            </div>


            <div class="note">

                The system first performs face detection and
                preprocessing. A MobileNetV2-based CNN is then
                used for feature extraction. The extracted feature
                vectors are classified using a Support Vector
                Machine (SVM).

                <br><br>

                The reported model score is derived from the SVM
                decision function. It should not be interpreted as
                a calibrated probability.

            </div>

        </section>


        <!-- RESPONSIBLE AI -->

        <section class="section">

            <div class="section-heading">

                <div class="section-number">
                    08
                </div>

                <div>

                    <h2>
                        Responsible AI Notice
                    </h2>

                    <p>
                        Important interpretation guidance
                    </p>

                </div>

            </div>


            <div class="note">

                This system is intended as an academic and research
                demonstration for identifying potentially manipulated
                media.

                <br><br>

                A model prediction should not be treated as definitive
                forensic proof. Results may be affected by image quality,
                lighting, face visibility, compression, manipulation type,
                and differences between training and real-world media.

            </div>

        </section>


    </main>


    <!-- =================================================
         FOOTER
         ================================================= -->

    <footer class="footer">

        <strong>
            DeepGuard AI
        </strong>

        <br>

        AI-Based Deepfake Detection System

        <br>

        CNN Feature Extraction + SVM Classification

        <br><br>

        This report was generated automatically by the
        DeepGuard AI detection application.

    </footer>


</div>


</body>

</html>
"""


    # -----------------------------------------------------
    # CREATE TEMPORARY REPORT
    # -----------------------------------------------------

    report_filename = (
        "deepfake_detection_report.html"
    )

    report_path = os.path.join(
        tempfile.gettempdir(),
        report_filename
    )


    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report_file:

        report_file.write(
            report_html
        )


    return FileResponse(
        path=report_path,
        media_type="text/html",
        filename=report_filename
    )