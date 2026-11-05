/* =========================================================
   DEEPGUARD AI — FRONTEND APPLICATION
   ========================================================= */

document.addEventListener("DOMContentLoaded", () => {

    /* =====================================================
       ELEMENTS
       ===================================================== */

    const navItems =
        document.querySelectorAll(".nav-item");

    const sections =
        document.querySelectorAll(".page-section");


    const heroAnalyzeBtn =
        document.getElementById("heroAnalyzeBtn");


    const fileInput =
        document.getElementById("fileInput");


    const dropZone =
        document.getElementById("dropZone");


    const analyzeButton =
        document.getElementById("analyzeButton");


    const selectedFile =
        document.getElementById("selectedFile");


    const fileName =
        document.getElementById("fileName");


    const fileSize =
        document.getElementById("fileSize");


    const removeFile =
        document.getElementById("removeFile");


    const mediaPreview =
        document.getElementById("mediaPreview");


    const previewContainer =
        document.getElementById("previewContainer");


    const previewRemove =
        document.getElementById("previewRemove");

    const forensicDetails =
        document.getElementById("forensicDetails");


    const resultEmpty =
        document.getElementById("resultEmpty");


    const analysisPipeline =
        document.getElementById("analysisPipeline");


    const resultContent =
        document.getElementById("resultContent");


    const resultError =
        document.getElementById("resultError");


    const resultStatus =
        document.getElementById("resultStatus");


    const resultLabel =
        document.getElementById("resultLabel");


    const resultMessage =
        document.getElementById("resultMessage");


    const resultBadge =
        document.getElementById("resultBadge");


    const resultBadgeIcon =
        document.getElementById("resultBadgeIcon");


    const confidenceValue =
        document.getElementById("confidenceValue");


    const confidenceBar =
        document.getElementById("confidenceBar");


    const videoStats =
        document.getElementById("videoStats");


    const framesAnalyzed =
        document.getElementById("framesAnalyzed");


    const faceDetection =
        document.getElementById("faceDetection");


    const realFrames =
        document.getElementById("realFrames");


    const fakeFrames =
        document.getElementById("fakeFrames");

    const reviewFrames =
        document.getElementById("reviewFrames");


    const frameDistribution =
        document.getElementById("frameDistribution");


    const fakePercentage =
        document.getElementById("fakePercentage");


    const realPart =
        document.getElementById("realPart");


    const fakePart =
        document.getElementById("fakePart");

    const reviewPart =
        document.getElementById("reviewPart");


    const analysisNote =
        document.getElementById("analysisNote");


    const errorTitle =
        document.getElementById("errorTitle");


    const errorMessage =
        document.getElementById("errorMessage");


    /* =====================================================
       TRY AGAIN BUTTONS
       ===================================================== */

    const tryAgainButton =
        document.getElementById("tryAgainButton");


    const errorTryAgainButton =
        document.getElementById("errorTryAgainButton");


    /* =====================================================
       DOWNLOAD REPORT
       ===================================================== */

    const downloadReportButton =
        document.getElementById(
            "downloadReportButton"
        );


    const pipelineTitle =
        document.getElementById("pipelineTitle");


    const pipelineDescription =
        document.getElementById(
            "pipelineDescription"
        );


    const pipelineProgressBar =
        document.getElementById(
            "pipelineProgressBar"
        );


    const stepFace =
        document.getElementById("stepFace");


    const stepCNN =
        document.getElementById("stepCNN");


    const stepSVM =
        document.getElementById("stepSVM");


    const stepFinal =
        document.getElementById("stepFinal");


    /* =====================================================
       STATE
       ===================================================== */

    let currentFile = null;

    let analysisTimer = null;

    let currentPipelineStep = 0;

    let latestDetectionResult = null;

    let previewObjectURL = null;

    let fingerprintRequestId = 0;


    /* =====================================================
       NAVIGATION
       ===================================================== */

    navItems.forEach((item) => {

        item.addEventListener(
            "click",
            () => {

                const targetSection =
                    item.dataset.section;


                navigateTo(
                    targetSection
                );

            }
        );

    });


    function navigateTo(sectionId) {

        navItems.forEach((item) => {

            item.classList.toggle(
                "active",
                item.dataset.section ===
                sectionId
            );

        });


        sections.forEach((section) => {

            section.classList.toggle(
                "active",
                section.id === sectionId
            );

        });


        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });

    }


    /* =====================================================
       HERO BUTTON
       ===================================================== */

    if (heroAnalyzeBtn) {

        heroAnalyzeBtn.addEventListener(
            "click",
            () => {

                navigateTo(
                    "analyzer"
                );


                setTimeout(() => {

                    if (fileInput) {

                        fileInput.click();

                    }

                }, 350);

            }
        );

    }


    /* =====================================================
       FILE INPUT
       ===================================================== */

    if (fileInput) {

        fileInput.addEventListener(
            "change",
            (event) => {

                const file =
                    event.target.files[0];


                if (file) {

                    handleFile(file);

                }

            }
        );

    }


    /* =====================================================
       DRAG & DROP
       ===================================================== */

    if (dropZone) {

        dropZone.addEventListener(
            "dragover",
            (event) => {

                event.preventDefault();

                dropZone.classList.add(
                    "dragover"
                );

            }
        );


        dropZone.addEventListener(
            "dragleave",
            () => {

                dropZone.classList.remove(
                    "dragover"
                );

            }
        );


        dropZone.addEventListener(
            "drop",
            (event) => {

                event.preventDefault();

                dropZone.classList.remove(
                    "dragover"
                );


                const file =
                    event.dataTransfer
                        .files[0];


                if (file) {

                    handleFile(file);

                }

            }
        );


        dropZone.addEventListener(
            "click",
            () => {

                if (fileInput) {

                    fileInput.click();

                }

            }
        );

    }


    /* =====================================================
       HANDLE FILE
       ===================================================== */

    function handleFile(file) {

        if (!isSupportedFile(file)) {

            showError(
                "Unsupported file",
                "Please upload a supported image or video file."
            );

            return;

        }


        currentFile = file;


        updateSelectedFile(
            file
        );


        createMediaPreview(
            file
        );


        resetResult();


        analyzeButton.disabled =
            false;

    }


    /* =====================================================
       SUPPORTED FILE CHECK
       ===================================================== */

    function isSupportedFile(file) {

        const allowedTypes = [

            "image/jpeg",
            "image/png",
            "image/webp",

            "video/mp4",
            "video/avi",
            "video/quicktime",
            "video/x-matroska",
            "video/webm"

        ];


        const extension =
            file.name
                .split(".")
                .pop()
                .toLowerCase();


        const allowedExtensions = [

            "jpg",
            "jpeg",
            "png",
            "webp",

            "mp4",
            "avi",
            "mov",
            "mkv",
            "webm"

        ];


        return (
            allowedTypes.includes(
                file.type
            ) ||
            allowedExtensions.includes(
                extension
            )
        );

    }


    /* =====================================================
       FILE DETAILS
       ===================================================== */

    function updateSelectedFile(file) {

        fileName.textContent =
            file.name;


        fileSize.textContent =
            formatFileSize(
                file.size
            );


        selectedFile.classList.add(
            "visible"
        );

    }


    function formatFileSize(bytes) {

        if (bytes === 0) {

            return "0 Bytes";

        }


        const units = [
            "Bytes",
            "KB",
            "MB",
            "GB"
        ];


        const index =
            Math.floor(
                Math.log(bytes) /
                Math.log(1024)
            );


        const size =
            bytes /
            Math.pow(
                1024,
                index
            );


        return (
            size.toFixed(2) +
            " " +
            units[index]
        );

    }


    /* =====================================================
       MEDIA PREVIEW
       ===================================================== */

    function createMediaPreview(file) {

        if (previewObjectURL) {

            URL.revokeObjectURL(
                previewObjectURL
            );

        }


        previewContainer.innerHTML =
            "";


        previewObjectURL =
            URL.createObjectURL(
                file
            );

        const objectURL =
            previewObjectURL;

        const mediaKind =
            getMediaKind(file);


        if (
            mediaKind === "image"
        ) {

            const image =
                document.createElement(
                    "img"
                );


            image.src =
                objectURL;


            image.alt =
                "Uploaded media preview";

            image.className =
                "preview-media";


            image.addEventListener(
                "load",
                () => {

                    document.getElementById(
                        "mediaDimensions"
                    ).textContent =
                        `${image.naturalWidth} × ${image.naturalHeight} px`;

                },
                { once: true }
            );


            previewContainer.appendChild(
                image
            );

        }

        else if (
            mediaKind === "video"
        ) {

            const video =
                document.createElement(
                    "video"
                );


            video.src =
                objectURL;


            video.controls =
                true;


            video.preload =
                "metadata";


            video.playsInline =
                true;

            video.className =
                "preview-media";


            video.addEventListener(
                "loadedmetadata",
                () => {

                    document.getElementById(
                        "mediaDimensions"
                    ).textContent =
                        `${video.videoWidth} × ${video.videoHeight} px`;

                    document.getElementById(
                        "mediaDuration"
                    ).textContent =
                        formatMediaDuration(
                            video.duration
                        );

                },
                { once: true }
            );


            previewContainer.appendChild(
                video
            );

        }


        mediaPreview.classList.add(
            "visible"
        );

        forensicDetails.classList.add(
            "visible"
        );


        updateForensicDetails(
            file,
            mediaKind
        );

    }


    function getMediaKind(file) {

        const extension =
            file.name
                .split(".")
                .pop()
                .toLowerCase();


        if (
            file.type.startsWith("image/") ||
            ["jpg", "jpeg", "png", "webp"].includes(extension)
        ) {

            return "image";

        }


        if (
            file.type.startsWith("video/") ||
            ["mp4", "avi", "mov", "mkv", "webm"].includes(extension)
        ) {

            return "video";

        }


        return "unknown";

    }


    function formatMediaDuration(seconds) {

        if (!Number.isFinite(seconds)) {

            return "Unavailable";

        }


        const minutes =
            Math.floor(seconds / 60);

        const remainingSeconds =
            Math.floor(seconds % 60);


        return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;

    }


    async function updateForensicDetails(file, mediaKind) {

        const requestId =
            ++fingerprintRequestId;

        const dimensions =
            document.getElementById("mediaDimensions");

        const duration =
            document.getElementById("mediaDuration");

        const fingerprint =
            document.getElementById("mediaFingerprint");


        dimensions.textContent =
            "Reading media…";

        duration.textContent =
            mediaKind === "video"
                ? "Reading media…"
                : "Not applicable";

        fingerprint.textContent =
            "Calculating locally…";


        if (
            file.size > 100 * 1024 * 1024
        ) {

            fingerprint.textContent =
                "Skipped for files over 100 MB";

            return;

        }


        if (
            !window.crypto ||
            !window.crypto.subtle
        ) {

            fingerprint.textContent =
                "Unavailable in this browser";

            return;

        }


        try {

            const fileBuffer =
                await file.arrayBuffer();

            const hashBuffer =
                await window.crypto.subtle.digest(
                    "SHA-256",
                    fileBuffer
                );


            if (requestId !== fingerprintRequestId) {

                return;

            }


            fingerprint.textContent =
                Array.from(
                    new Uint8Array(hashBuffer),
                    (byte) => byte.toString(16).padStart(2, "0")
                ).join("");

            if (currentFile === file && latestDetectionResult) {

                latestDetectionResult.file_fingerprint =
                    fingerprint.textContent;

                document.getElementById(
                    "resultFingerprint"
                ).textContent =
                    fingerprint.textContent;

            }

        }

        catch (error) {

            if (requestId !== fingerprintRequestId) {

                return;

            }


            fingerprint.textContent =
                "Could not calculate fingerprint";

            console.error(
                "Unable to calculate the local media fingerprint.",
                error
            );

        }

    }


    /* =====================================================
       REMOVE FILE
       ===================================================== */

    if (removeFile) {

        removeFile.addEventListener(
            "click",
            () => {

                clearFile();

            }
        );

    }


    if (previewRemove) {

        previewRemove.addEventListener(
            "click",
            () => {

                clearFile();

            }
        );

    }


    function clearFile() {

        currentFile =
            null;


        latestDetectionResult =
            null;

        fingerprintRequestId += 1;


        if (previewObjectURL) {

            URL.revokeObjectURL(
                previewObjectURL
            );

            previewObjectURL =
                null;

        }


        if (fileInput) {

            fileInput.value =
                "";

        }


        fileName.textContent =
            "filename";


        fileSize.textContent =
            "0 KB";


        selectedFile.classList.remove(
            "visible"
        );


        mediaPreview.classList.remove(
            "visible"
        );

        forensicDetails.classList.remove(
            "visible"
        );


        previewContainer.innerHTML =
            "";

        document.getElementById(
            "mediaDimensions"
        ).textContent =
            "—";

        document.getElementById(
            "mediaDuration"
        ).textContent =
            "—";

        document.getElementById(
            "mediaFingerprint"
        ).textContent =
            "—";


        analyzeButton.disabled =
            true;


        resetResult();

    }


    /* =====================================================
       ANALYZE BUTTON
       ===================================================== */

    if (analyzeButton) {

        analyzeButton.addEventListener(
            "click",
            async () => {

                if (!currentFile) {

                    showError(
                        "No media selected",
                        "Please select an image or video first."
                    );

                    return;

                }


                await analyzeMedia();

            }
        );

    }


    /* =====================================================
       ANALYZE MEDIA
       ===================================================== */

    async function analyzeMedia() {

        resetResult();


        startAnalysisPipeline();


        analyzeButton.disabled =
            true;


        const buttonText =
            analyzeButton.querySelector(
                "span:first-child"
            );


        if (buttonText) {

            buttonText.textContent =
                "Analyzing...";

        }


        const formData =
            new FormData();


        formData.append(
            "file",
            currentFile
        );


        const isVideo =
            currentFile.type.startsWith(
                "video/"
            ) ||
            isVideoExtension(
                currentFile.name
            );


        const endpoint =
            isVideo
                ? "/predict/video"
                : "/predict/image";


        try {

            const response =
                await fetch(
                    endpoint,
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            finishAnalysisPipeline();


            await wait(500);


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    data.message ||
                    "The server could not process the media."
                );

            }


            if (
                data.success === false &&
                data.status ===
                    "insufficient_face_data"
            ) {

                showInsufficientFaceResult(
                    data
                );

                return;

            }


            if (!data.success) {

                throw new Error(
                    data.message ||
                    "Analysis could not be completed."
                );

            }


            /* =================================================
               SAVE LATEST DETECTION RESULT
               ================================================= */

            latestDetectionResult = {

                ...data,

                filename:
                    currentFile.name,

                media_type:
                    isVideo
                        ? "video"
                        : "image",

                file_size:
                    currentFile.size,

                file_type:
                    currentFile.type ||
                    currentFile.name.split(".").pop().toLowerCase(),

                file_fingerprint:
                    /^[a-f0-9]{64}$/i.test(
                        document.getElementById(
                            "mediaFingerprint"
                        ).textContent
                    )
                        ? document.getElementById(
                            "mediaFingerprint"
                        ).textContent
                        : null

            };


            /* =================================================
               SHOW RESULT
               ================================================= */

            showResult(
                latestDetectionResult,
                isVideo
            );

        }

        catch (error) {

            stopAnalysisPipeline();


            showError(
                "Analysis failed",
                error.message ||
                "Unable to connect to the AI analysis service."
            );

        }

        finally {

            analyzeButton.disabled =
                false;


            if (buttonText) {

                buttonText.textContent =
                    "Analyze with AI";

            }

        }

    }


    /* =====================================================
       VIDEO EXTENSION
       ===================================================== */

    function isVideoExtension(
        filename
    ) {

        const extension =
            filename
                .split(".")
                .pop()
                .toLowerCase();


        return [

            "mp4",
            "avi",
            "mov",
            "mkv",
            "webm"

        ].includes(
            extension
        );

    }


    /* =====================================================
       ANALYSIS PIPELINE
       ===================================================== */

    function startAnalysisPipeline() {

        resultEmpty.style.display =
            "none";


        resultContent.classList.remove(
            "visible"
        );


        resultError.classList.remove(
            "visible"
        );


        analysisPipeline.classList.add(
            "active"
        );


        resultStatus.textContent =
            "ANALYZING";


        resultStatus.style.background =
            "#f1efff";


        resultStatus.style.color =
            "#6d5dfc";


        resetPipelineSteps();


        currentPipelineStep =
            0;


        pipelineTitle.textContent =
            "Starting AI analysis";


        pipelineDescription.textContent =
            "Preparing your media for forensic analysis...";


        pipelineProgressBar.style.width =
            "0%";


        runPipelineAnimation();

    }


    function runPipelineAnimation() {

        const steps = [

            {

                element:
                    stepFace,

                title:
                    "Detecting faces",

                description:
                    "Locating facial regions in the uploaded media...",

                progress:
                    20

            },

            {

                element:
                    stepCNN,

                title:
                    "Extracting CNN features",

                description:
                    "MobileNetV2 is extracting visual representations...",

                progress:
                    50

            },

            {

                element:
                    stepSVM,

                title:
                    "Running SVM classification",

                description:
                    "The RBF SVM is evaluating authenticity patterns...",

                progress:
                    75

            },

            {

                element:
                    stepFinal,

                title:
                    "Generating final decision",

                description:
                    "Aggregating the model analysis into a final result...",

                progress:
                    92

            }

        ];


        let stepIndex =
            0;


        function activateNextStep() {

            if (
                stepIndex >=
                steps.length
            ) {

                return;

            }


            if (
                stepIndex > 0
            ) {

                steps[
                    stepIndex - 1
                ]
                    .element
                    .classList
                    .remove(
                        "active"
                    );


                steps[
                    stepIndex - 1
                ]
                    .element
                    .classList
                    .add(
                        "completed"
                    );

            }


            const step =
                steps[
                    stepIndex
                ];


            step.element.classList.add(
                "active"
            );


            pipelineTitle.textContent =
                step.title;


            pipelineDescription.textContent =
                step.description;


            pipelineProgressBar.style.width =
                step.progress +
                "%";


            stepIndex++;


            analysisTimer =
                setTimeout(
                    activateNextStep,
                    750
                );

        }


        activateNextStep();

    }


    function finishAnalysisPipeline() {

        if (analysisTimer) {

            clearTimeout(
                analysisTimer
            );


            analysisTimer =
                null;

        }


        const steps = [

            stepFace,
            stepCNN,
            stepSVM,
            stepFinal

        ];


        steps.forEach(
            (step) => {

                step.classList.remove(
                    "active"
                );


                step.classList.add(
                    "completed"
                );

            }
        );


        pipelineTitle.textContent =
            "Analysis complete";


        pipelineDescription.textContent =
            "The AI model has finished evaluating the media.";


        pipelineProgressBar.style.width =
            "100%";

    }


    function stopAnalysisPipeline() {

        if (analysisTimer) {

            clearTimeout(
                analysisTimer
            );


            analysisTimer =
                null;

        }


        analysisPipeline.classList.remove(
            "active"
        );

    }


    function resetPipelineSteps() {

        [

            stepFace,
            stepCNN,
            stepSVM,
            stepFinal

        ].forEach(
            (step) => {

                step.classList.remove(
                    "active",
                    "completed"
                );

            }
        );

    }


    /* =====================================================
       SHOW RESULT
       ===================================================== */

    function showResult(
        data,
        isVideo
    ) {

        stopAnalysisPipeline();


        resultEmpty.style.display =
            "none";


        resultError.classList.remove(
            "visible"
        );


        resultContent.classList.add(
            "visible"
        );


        const label =
            data.label ||
            "Unknown";

        const isNeedsReview =
            label.toLowerCase() === "needs review";

        resultStatus.textContent =
            isNeedsReview ? "NEEDS REVIEW" : "COMPLETE";

        resultStatus.style.background =
            isNeedsReview ? "#fff5e8" : "#e9faf3";

        resultStatus.style.color =
            isNeedsReview ? "#c27a16" : "#18a874";


        const scoreValue =
            data.model_score != null
                ? Number(data.model_score)
                : data.score != null
                    ? Number(data.score) * 100
                    : data.confidence != null
                        ? Number(data.confidence)
                        : null;


        const safeScore =
            scoreValue == null || !Number.isFinite(scoreValue)
                ? null
                : Math.max(0, Math.min(100, scoreValue));


        resultLabel.textContent =
            label;

        resultBadge.classList.remove(
            "fake",
            "review"
        );


        if (
            label.toLowerCase() ===
            "fake"
        ) {

            resultBadge.classList.add(
                "fake"
            );


            resultBadgeIcon.textContent =
                "!";


            resultMessage.textContent =
                "The model detected patterns associated with manipulated media.";

        }

        else if (isNeedsReview) {

            resultBadge.classList.add(
                "review"
            );

            resultBadgeIcon.textContent =
                "?";

            resultMessage.textContent =
                data.message ||
                "The available evidence is inconclusive. Manual review is recommended.";

        }

        else {

            resultBadgeIcon.textContent =
                "✓";


            resultMessage.textContent =
                "The model classified the analyzed media as real.";

        }


        confidenceValue.textContent =
            safeScore == null
                ? "Not available"
                : safeScore.toFixed(2) + "%";


        setTimeout(
            () => {

                confidenceBar.style.width =
                    (safeScore ?? 0) +
                    "%";

            },
            100
        );

        updateForensicResult(
            data,
            isVideo
        );


        /* =================================================
           VIDEO INFORMATION
           ================================================= */

        if (isVideo) {

            updateVideoStats(
                data
            );

        }

        else {

            videoStats.classList.remove(
                "visible"
            );


            frameDistribution.classList.remove(
                "visible"
            );

        }


        analysisNote.textContent =
            isNeedsReview
                ? "The available evidence is inconclusive. Manual review is recommended; the model score is not a calibrated probability."
                : "This result is an AI-assisted assessment and should be interpreted as an automated screening result rather than definitive forensic proof. The model score is not a calibrated probability.";

        /* SAVE SUCCESSFUL ANALYSIS TO DASHBOARD */

        saveDetectionToHistory(
            data,
            isVideo,
            currentFile
                ? currentFile.name
                : "Unknown file"
        );
    }


    function updateForensicResult(data, isVideo) {

        const faceCount =
            isVideo
                ? Number(data.face_detected_frames ?? 0)
                : data.face_detected
                    ? 1
                    : 0;

        const frameCount =
            isVideo
                ? Number(data.analyzed_frames ?? 0)
                : 1;

        const faceRate =
            isVideo
                ? Number(data.face_detection_percentage ?? 0)
                : faceCount * 100;

        const scoredFrameCount =
            Array.isArray(data.frame_results)
                ? data.frame_results.filter(
                    (frame) => frame.score != null
                ).length
                : 0;

        const hasModelScore =
            data.model_score != null ||
            data.score != null;

        const agreement =
            data.model_agreement_percentage;


        document.getElementById(
            "evidenceFace"
        ).textContent =
            isVideo
                ? `${faceCount}/${frameCount} sampled frames (${faceRate.toFixed(1)}%)`
                : faceCount
                    ? "Face detected"
                    : "No face detected";

        document.getElementById(
            "evidenceCNN"
        ).textContent =
            hasModelScore
                ? "Facial crop features extracted"
                : "Not available — no usable face crop";

        document.getElementById(
            "evidenceSVM"
        ).textContent =
            hasModelScore
                ? "Classification score produced"
                : "Not run — no usable features";

        document.getElementById(
            "evidenceAgreement"
        ).textContent =
            agreement != null
                ? `${Number(agreement).toFixed(1)}% across ${scoredFrameCount} scored frames`
                : "Not available for a single image";


        const faceStrength =
            faceCount === 0
                ? "Low"
                : isVideo
                    ? faceRate >= 80
                        ? "High"
                        : faceRate >= 50
                            ? "Medium"
                            : "Low"
                    : "Available (single image)";

        const agreementStrength =
            agreement == null
                ? "Not available"
                : agreement >= 80
                    ? "High"
                    : agreement >= 60
                        ? "Medium"
                        : "Low";

        const overallStrength =
            faceCount === 0 ||
            (isVideo && (faceCount < 3 || faceRate < 50))
                ? "Low"
                : data.label === "Needs Review"
                    ? "Low"
                    : isVideo && agreement != null && faceRate >= 80 && agreement >= 80
                        ? "High"
                        : "Medium";


        document.getElementById(
            "strengthFace"
        ).textContent =
            faceStrength;

        document.getElementById(
            "strengthAgreement"
        ).textContent =
            agreementStrength;

        document.getElementById(
            "strengthOverall"
        ).textContent =
            overallStrength;

        data.evidence_strength =
            overallStrength;

        data.face_coverage_strength =
            faceStrength;

        data.frame_agreement_strength =
            agreementStrength;


        const reviewReason =
            document.getElementById("reviewReason");

        reviewReason.hidden =
            data.label !== "Needs Review";

        reviewReason.textContent =
            data.label === "Needs Review"
                ? data.decision_reason ||
                    "The available evidence is inconclusive. Manual review is recommended."
                : "";


        const metadataValues = {
            resultFileName: data.filename,
            resultMediaType: data.media_type,
            resultFileSize: data.file_size != null
                ? formatFileSize(Number(data.file_size))
                : null,
            resultResolution: data.image_width && data.image_height
                ? `${data.image_width} × ${data.image_height} px`
                : data.video_width && data.video_height
                    ? `${data.video_width} × ${data.video_height} px`
                    : null,
            resultDuration: data.video_duration != null
                ? formatMediaDuration(Number(data.video_duration))
                : null,
            resultFps: data.video_fps != null
                ? `${Number(data.video_fps).toFixed(2)} FPS`
                : null,
            resultFramesAnalyzed: isVideo
                ? data.analyzed_frames
                : "1 image",
            resultFacesDetected: isVideo
                ? data.face_detected_frames
                : data.face_detected
                    ? "1 face detected"
                    : "0",
            resultRealFrames: isVideo ? data.real_frames : null,
            resultFakeFrames: isVideo ? data.fake_frames : null,
            resultSuspiciousFrames: isVideo ? data.suspicious_frames : null,
            resultMedianScore: data.median_score != null
                ? Number(data.median_score).toFixed(4)
                : data.score != null
                    ? Number(data.score).toFixed(4)
                    : null,
            resultFinalAssessment: data.label,
            resultFingerprint: data.file_fingerprint
        };


        Object.entries(metadataValues).forEach(
            ([id, value]) => {

                document.getElementById(id).textContent =
                    value == null || value === ""
                        ? "Not available"
                        : String(value);

            }
        );


        renderFrameTimeline(
            data.frame_results,
            isVideo
        );

    }


    function renderFrameTimeline(frameResults, isVideo) {

        const section =
            document.getElementById("frameInvestigationSection");

        const timeline =
            document.getElementById("frameTimeline");

        const inspection =
            document.getElementById("frameInspection");


        timeline.replaceChildren();
        inspection.hidden = true;
        section.style.display =
            isVideo ? "" : "none";


        if (!isVideo || !Array.isArray(frameResults)) {

            return;

        }


        frameResults.forEach(
            (frame) => {

                const button =
                    document.createElement("button");

                const timestamp =
                    frame.timestamp == null
                        ? "Time unavailable"
                        : formatMediaDuration(Number(frame.timestamp));

                const assessment =
                    frame.assessment || "Not analyzed";

                const score =
                    frame.model_score == null
                        ? "Score unavailable"
                        : `Model score ${Number(frame.model_score).toFixed(1)}%`;


                button.type = "button";
                button.className = "frame-tile";
                button.classList.add(
                    assessment === "Fake"
                        ? "assessment-fake"
                        : assessment === "Real"
                            ? "assessment-real"
                            : "assessment-review"
                );
                button.setAttribute(
                    "aria-label",
                    `Frame ${frame.frame_number}, ${timestamp}, ${assessment}`
                );

                if (frame.frame_image) {
                    const image =
                        document.createElement("img");

                    image.src = frame.frame_image;
                    image.alt = `Video frame ${frame.frame_number}`;
                    button.appendChild(image);
                }


                const title =
                    document.createElement("strong");

                title.textContent =
                    `Frame ${frame.frame_number}`;

                const time =
                    document.createElement("span");

                time.textContent =
                    timestamp;

                const result =
                    document.createElement("span");

                result.textContent =
                    `${assessment} · ${score}`;

                const face =
                    document.createElement("span");

                face.textContent =
                    frame.face_detected
                        ? "Face detected"
                        : "No face detected";

                const suspicion =
                    document.createElement("span");

                suspicion.textContent =
                    `Suspicion: ${frame.suspicion_level || "Unavailable"}`;

                button.append(
                    title,
                    time,
                    result,
                    face,
                    suspicion
                );

                button.addEventListener(
                    "click",
                    () => {

                        inspection.hidden = false;

                        const frameImage =
                            document.getElementById(
                                "investigatedFrameImage"
                            );

                        frameImage.src =
                            frame.frame_image || "";

                        frameImage.hidden =
                            !frame.frame_image;

                        const faceImage =
                            document.getElementById(
                                "investigatedFaceImage"
                            );

                        faceImage.src =
                            frame.face_crop || "";

                        faceImage.hidden =
                            !frame.face_crop;

                        document.getElementById(
                            "investigatedFrameSummary"
                        ).textContent =
                            `${timestamp} · Frame ${frame.frame_number} · ` +
                            `${frame.face_detected ? "Face detected" : "No face detected"} · ` +
                            `${assessment} · ${score} · Suspicion: ${frame.suspicion_level || "Unavailable"}`;

                    }
                );

                timeline.appendChild(button);

            }
        );

    }


    /* =====================================================
       VIDEO STATS
       ===================================================== */

    function updateVideoStats(
        data
    ) {

        videoStats.classList.add(
            "visible"
        );


        frameDistribution.classList.add(
            "visible"
        );


        framesAnalyzed.textContent =
            data.analyzed_frames ??
            "—";


        faceDetection.textContent =
            data.face_detection_percentage !=
            null
                ? data.face_detection_percentage +
                  "%"
                : "—";


        realFrames.textContent =
            data.real_frames ??
            "—";


        fakeFrames.textContent =
            data.fake_frames ??
            "—";

        reviewFrames.textContent =
            data.suspicious_frames ??
            "—";


        const fakePercent =
            Number(
                data.fake_frame_percentage ??
                0
            );

        const realPercent =
            Number(
                data.real_frame_percentage ??
                100 - fakePercent -
                    Number(data.suspicious_frame_percentage ?? 0)
            );

        const reviewPercent =
            Number(
                data.suspicious_frame_percentage ??
                0
            );


        const safeFakePercent =
            Math.max(
                0,
                Math.min(
                    100,
                    fakePercent
                )
            );


        const safeRealPercent =
            Math.max(
                0,
                Math.min(
                    100,
                    realPercent
                )
            );

        const safeReviewPercent =
            Math.max(
                0,
                Math.min(
                    100,
                    reviewPercent
                )
            );


        fakePercentage.textContent =
            safeFakePercent.toFixed(1) +
            "% Fake";


        setTimeout(
            () => {

                realPart.style.width =
                    safeRealPercent +
                    "%";


                fakePart.style.width =
                    safeFakePercent +
                    "%";

                reviewPart.style.width =
                    safeReviewPercent +
                    "%";

            },
            150
        );

    }


    /* =====================================================
       INSUFFICIENT FACE DATA
       ===================================================== */

    function showInsufficientFaceResult(
        data
    ) {

        stopAnalysisPipeline();


        latestDetectionResult =
            null;


        resultEmpty.style.display =
            "none";


        resultContent.classList.remove(
            "visible"
        );


        resultError.classList.add(
            "visible"
        );


        resultStatus.textContent =
            "INSUFFICIENT DATA";


        resultStatus.style.background =
            "#fff5e8";


        resultStatus.style.color =
            "#c27a16";


        errorTitle.textContent =
            "Not enough facial data";


        errorMessage.textContent =
            data.message ||
            "The video does not contain enough clearly detectable facial frames for a reliable prediction.";


        analysisNote.textContent =
            "The system intentionally does not generate a Real/Fake prediction when too few facial frames are available.";

    }


    /* =====================================================
       ERROR
       ===================================================== */

    function showError(
        title,
        message
    ) {

        stopAnalysisPipeline();


        latestDetectionResult =
            null;


        resultEmpty.style.display =
            "none";


        resultContent.classList.remove(
            "visible"
        );


        resultError.classList.add(
            "visible"
        );


        resultStatus.textContent =
            "ERROR";


        resultStatus.style.background =
            "#fff0f0";


        resultStatus.style.color =
            "#e05252";


        errorTitle.textContent =
            title;


        errorMessage.textContent =
            message;

    }


    /* =====================================================
       TRY AGAIN — SUCCESS RESULT
       ===================================================== */

    if (tryAgainButton) {

        tryAgainButton.addEventListener(
            "click",
            () => {

                resetResult();

            }
        );

    }


    /* =====================================================
       TRY AGAIN — ERROR RESULT
       ===================================================== */

    if (errorTryAgainButton) {

        errorTryAgainButton.addEventListener(
            "click",
            () => {

                resetResult();

            }
        );

    }


    /* =====================================================
       RESET RESULT
       ===================================================== */

    function resetResult() {

        if (analysisTimer) {

            clearTimeout(
                analysisTimer
            );


            analysisTimer =
                null;

        }


        /* Clear previous detection result */

        latestDetectionResult =
            null;


        analysisPipeline.classList.remove(
            "active"
        );


        resultContent.classList.remove(
            "visible"
        );


        resultError.classList.remove(
            "visible"
        );


        resultEmpty.style.display =
            "flex";


        resultStatus.textContent =
            "WAITING";


        resultStatus.style.background =
            "#f3f4f8";


        resultStatus.style.color =
            "#8b93a2";


        confidenceValue.textContent =
            "0%";


        confidenceBar.style.width =
            "0%";


        videoStats.classList.remove(
            "visible"
        );


        frameDistribution.classList.remove(
            "visible"
        );


        realPart.style.width =
            "100%";


        fakePart.style.width =
            "0%";

        reviewPart.style.width =
            "0%";

        reviewFrames.textContent =
            "—";


        framesAnalyzed.textContent =
            "—";


        faceDetection.textContent =
            "—";


        realFrames.textContent =
            "—";


        fakeFrames.textContent =
            "—";


        fakePercentage.textContent =
            "0% Fake";


        resetPipelineSteps();

    }


    /* =====================================================
       UTILITY
       ===================================================== */

    function wait(ms) {

        return new Promise(
            (resolve) => {

                setTimeout(
                    resolve,
                    ms
                );

            }
        );

    }


    /* =====================================================
       DOWNLOAD DETECTION REPORT
       ===================================================== */

    if (downloadReportButton) {

        downloadReportButton.addEventListener(
            "click",
            async () => {

                if (!latestDetectionResult) {

                    showError(
                        "No detection result",
                        "Please analyze an image or video before downloading the report."
                    );

                    return;

                }


                try {

                    downloadReportButton.disabled =
                        true;


                    const originalText =
                        downloadReportButton
                            .textContent;


                    downloadReportButton.textContent =
                        "Generating Report...";


                    const response =
                        await fetch(
                            "/report",
                            {
                                method:
                                    "POST",

                                headers: {
                                    "Content-Type":
                                        "application/json"
                                },

                                body:
                                    JSON.stringify(
                                        {
                                            ...latestDetectionResult,
                                            frame_results:
                                                Array.isArray(
                                                    latestDetectionResult.frame_results
                                                )
                                                    ? latestDetectionResult.frame_results.map(
                                                        ({
                                                            frame_image,
                                                            face_crop,
                                                            ...frame
                                                        }) => frame
                                                    )
                                                    : []
                                        }
                                    )

                            }
                        );


                    if (!response.ok) {

                        let errorMessage =
                            "Unable to generate the detection report.";


                        try {

                            const errorData =
                                await response.json();


                            errorMessage =
                                errorData.detail ||
                                errorData.message ||
                                errorMessage;

                        }

                        catch (_) {

                            /* Keep default message */

                        }


                        throw new Error(
                            errorMessage
                        );

                    }


                    const blob =
                        await response.blob();


                    const downloadURL =
                        window.URL.createObjectURL(
                            blob
                        );


                    const link =
                        document.createElement(
                            "a"
                        );


                    link.href =
                        downloadURL;


                    link.download =
                        "deepfake_detection_report.html";


                    document.body.appendChild(
                        link
                    );


                    link.click();


                    link.remove();


                    window.URL.revokeObjectURL(
                        downloadURL
                    );


                    downloadReportButton.textContent =
                        "Report Downloaded ✓";


                    setTimeout(
                        () => {

                            downloadReportButton.textContent =
                                originalText;


                            downloadReportButton.disabled =
                                false;

                        },
                        2000
                    );

                }

                catch (error) {

                    console.error(
                        "Report generation error:",
                        error
                    );


                    showError(
                        "Report generation failed",
                        error.message ||
                        "Unable to generate the detection report."
                    );


                    downloadReportButton.disabled =
                        false;

                }

            }
        );

    }

        /* =========================================================
       DEEPGUARD AI — DASHBOARD ANALYTICS
       ========================================================= */

    const DASHBOARD_HISTORY_KEY =
        "deepguard_detection_history";

    const dashboardSection = document.getElementById("analytics");


    /* =========================================================
       HISTORY STORAGE
       ========================================================= */

    function getDetectionHistory() {
        try {
            const stored =
                localStorage.getItem(
                    DASHBOARD_HISTORY_KEY
                );

            if (!stored) {
                return [];
            }

            const parsed =
                JSON.parse(stored);

            return Array.isArray(parsed)
                ? parsed
                : [];

        } catch (error) {
            console.error(
                "Unable to load detection history:",
                error
            );

            return [];
        }
    }


    function saveDetectionToHistory(
        data,
        isVideo,
        filename
    ) {

        const label =
            data.label || "Unknown";

        const confidenceValue =
            data.confidence ??
            data.score;

        const confidence =
            confidenceValue == null
                ? null
                : Number(confidenceValue);

        const history =
            getDetectionHistory();

        const entry = {
            id:
                Date.now() +
                "-" +
                Math.random()
                    .toString(36)
                    .slice(2, 8),

            filename:
                filename ||
                "Unknown file",

            media_type:
                isVideo
                    ? "Video"
                    : "Image",

            label:
                label,

            model_score:
                data.model_score != null
                    ? Number(data.model_score)
                    : data.score != null
                        ? Number(data.score) * 100
                        : null,

            confidence:
                confidence == null || !Number.isFinite(confidence)
                    ? null
                    : Math.max(
                        0,
                        Math.min(
                            100,
                            confidence
                        )
                    ),

            timestamp:
                new Date().toISOString(),

            analyzed_frames:
                data.analyzed_frames ??
                (isVideo ? null : 1),

            face_detected_frames:
                data.face_detected_frames ??
                (isVideo
                    ? null
                    : data.face_detected
                        ? 1
                        : 0),

            face_detection_percentage:
                data.face_detection_percentage ??
                (isVideo
                    ? null
                    : data.face_detected
                        ? 100
                        : 0),

            real_frames:
                data.real_frames ?? null,

            fake_frames:
                data.fake_frames ?? null,

            real_like_frames:
                data.real_like_frames ?? data.real_frames ?? null,

            fake_like_frames:
                data.fake_like_frames ?? data.fake_frames ?? null,

            suspicious_frames:
                data.suspicious_frames ??
                (Array.isArray(data.frame_results)
                    ? data.frame_results.filter(
                        (frame) => frame.assessment === "Needs Review"
                    ).length
                    : 0),

            fake_frame_percentage:
                isVideo
                    ? (
                        data.fake_frame_percentage ??
                        null
                    )
                    : null
        };


        history.unshift(entry);


        /*
         * Keep the latest 100 analyses.
         */
        const trimmedHistory =
            history.slice(0, 100);


        try {

            localStorage.setItem(
                DASHBOARD_HISTORY_KEY,
                JSON.stringify(
                    trimmedHistory
                )
            );

        } catch (error) {

            console.error(
                "Unable to save detection history:",
                error
            );

        }


        updateDashboard();
    }


    /* =========================================================
       FORMAT HELPERS
       ========================================================= */

    function formatDashboardDate(
        timestamp
    ) {

        if (!timestamp) {
            return "—";
        }

        const date =
            new Date(timestamp);

        if (
            Number.isNaN(
                date.getTime()
            )
        ) {
            return "—";
        }

        return date.toLocaleString(
            undefined,
            {
                day: "2-digit",
                month: "short",
                year: "numeric",
                hour: "2-digit",
                minute: "2-digit"
            }
        );
    }


    function escapeDashboardHTML(
        value
    ) {

        return String(value ?? "")
            .replace(
                /&/g,
                "&amp;"
            )
            .replace(
                /</g,
                "&lt;"
            )
            .replace(
                />/g,
                "&gt;"
            )
            .replace(
                /"/g,
                "&quot;"
            )
            .replace(
                /'/g,
                "&#039;"
            );
    }


    /* =========================================================
       DASHBOARD STYLES
       ========================================================= */

    function injectDashboardStyles() {

        if (
            document.getElementById(
                "deepguardDashboardStyles"
            )
        ) {
            return;
        }


        const style =
            document.createElement(
                "style"
            );

        style.id =
            "deepguardDashboardStyles";


        style.textContent = `

            /* =========================================
               DASHBOARD ANALYTICS
               ========================================= */

            .deepguard-dashboard {
                margin-top: 34px;
                width: 100%;
            }


            .dashboard-heading {
                display: flex;
                align-items: flex-end;
                justify-content: space-between;
                gap: 20px;
                margin-bottom: 20px;
            }


            .dashboard-heading h2 {
                margin: 0;
                font-size: 25px;
                font-weight: 800;
                color: #202437;
                letter-spacing: -0.5px;
            }


            .dashboard-heading p {
                margin: 7px 0 0;
                color: #858ba0;
                font-size: 14px;
                line-height: 1.6;
            }


            .dashboard-heading-actions {
                display: flex;
                gap: 10px;
                align-items: center;
            }


            .dashboard-refresh-button,
            .dashboard-clear-button {
                border: 1px solid #e6e7f0;
                background: #ffffff;
                color: #60677c;
                border-radius: 11px;
                padding: 10px 15px;
                font-size: 13px;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.2s ease;
            }


            .dashboard-refresh-button:hover {
                border-color: #7464f5;
                color: #6657df;
                transform: translateY(-1px);
            }


            .dashboard-clear-button:hover {
                border-color: #ef8a8a;
                color: #d95454;
                transform: translateY(-1px);
            }


            /* =========================================
               STAT CARDS
               ========================================= */

            .dashboard-stat-grid {
                display: grid;
                grid-template-columns:
                    repeat(5, minmax(0, 1fr));
                gap: 16px;
                margin-bottom: 20px;
            }


            .dashboard-stat-card {
                position: relative;
                overflow: hidden;
                background: #ffffff;
                border: 1px solid #ececf3;
                border-radius: 18px;
                padding: 20px;
                box-shadow:
                    0 8px 25px
                    rgba(50, 44, 100, 0.045);
                transition:
                    transform 0.25s ease,
                    box-shadow 0.25s ease;
            }


            .dashboard-stat-card:hover {
                transform: translateY(-3px);
                box-shadow:
                    0 14px 32px
                    rgba(50, 44, 100, 0.08);
            }


            .dashboard-stat-card::after {
                content: "";
                position: absolute;
                width: 75px;
                height: 75px;
                border-radius: 50%;
                right: -28px;
                top: -28px;
                background:
                    rgba(109, 93, 252, 0.06);
            }


            .dashboard-stat-icon {
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 12px;
                background: #f3f1ff;
                color: #6d5dfc;
                font-size: 18px;
                margin-bottom: 14px;
            }


            .dashboard-stat-label {
                font-size: 12px;
                color: #8a90a2;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.6px;
            }


            .dashboard-stat-value {
                margin-top: 5px;
                font-size: 29px;
                font-weight: 850;
                color: #202437;
            }


            .dashboard-stat-subtext {
                margin-top: 4px;
                color: #999eaf;
                font-size: 12px;
            }


            .dashboard-stat-card.real
            .dashboard-stat-icon {
                background: #eafaf3;
                color: #18a874;
            }


            .dashboard-stat-card.fake
            .dashboard-stat-icon {
                background: #fff0f0;
                color: #df5b5b;
            }


            .dashboard-stat-card.confidence
            .dashboard-stat-icon {
                background: #fff6e8;
                color: #c7851a;
            }

            .dashboard-stat-card.review
            .dashboard-stat-icon {
                background: #fff5e5;
                color: #c27a16;
            }


            /* =========================================
               CHART GRID
               ========================================= */

            .dashboard-chart-grid {
                display: grid;
                grid-template-columns:
                    minmax(0, 0.9fr)
                    minmax(0, 1.5fr);
                gap: 20px;
                margin-bottom: 20px;
            }


            .dashboard-panel {
                background: #ffffff;
                border: 1px solid #ececf3;
                border-radius: 18px;
                padding: 22px;
                box-shadow:
                    0 8px 25px
                    rgba(50, 44, 100, 0.045);
            }


            .dashboard-panel-header {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 15px;
                margin-bottom: 20px;
            }


            .dashboard-panel-title {
                margin: 0;
                font-size: 16px;
                font-weight: 800;
                color: #262a3d;
            }


            .dashboard-panel-description {
                margin: 5px 0 0;
                color: #9297a8;
                font-size: 12px;
            }


            /* =========================================
               DONUT CHART
               ========================================= */

            .dashboard-donut-area {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 30px;
                min-height: 210px;
            }


            .dashboard-donut {
                width: 165px;
                height: 165px;
                border-radius: 50%;
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
                background:
                    conic-gradient(
                        #18a874 0deg,
                        #18a874 0deg,
                        #df5b5b 0deg,
                        #df5b5b 360deg
                    );
            }


            .dashboard-donut::after {
                content: "";
                position: absolute;
                width: 112px;
                height: 112px;
                background: #ffffff;
                border-radius: 50%;
            }


            .dashboard-donut-center {
                position: relative;
                z-index: 2;
                text-align: center;
            }


            .dashboard-donut-number {
                display: block;
                font-size: 28px;
                font-weight: 850;
                color: #22263a;
            }


            .dashboard-donut-label {
                display: block;
                margin-top: 2px;
                color: #9297a8;
                font-size: 11px;
                font-weight: 700;
            }


            .dashboard-legend {
                display: flex;
                flex-direction: column;
                gap: 14px;
            }


            .dashboard-legend-item {
                display: flex;
                align-items: center;
                gap: 9px;
                min-width: 110px;
            }


            .dashboard-legend-dot {
                width: 10px;
                height: 10px;
                border-radius: 50%;
                flex-shrink: 0;
            }


            .dashboard-legend-dot.real {
                background: #18a874;
            }


            .dashboard-legend-dot.fake {
                background: #df5b5b;
            }

            .dashboard-legend-dot.review {
                background: #e1a33a;
            }


            .dashboard-legend-text {
                display: flex;
                flex-direction: column;
            }


            .dashboard-legend-name {
                color: #72788b;
                font-size: 12px;
                font-weight: 700;
            }


            .dashboard-legend-value {
                color: #25293b;
                font-size: 18px;
                font-weight: 850;
            }


            /* =========================================
               BAR CHART
               ========================================= */

            .dashboard-bars {
                min-height: 210px;
                display: flex;
                align-items: flex-end;
                gap: 14px;
                padding:
                    15px 5px 0;
            }


            .dashboard-bar-item {
                flex: 1;
                height: 190px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: flex-end;
                min-width: 0;
            }


            .dashboard-bar-value {
                color: #686e82;
                font-size: 10px;
                font-weight: 800;
                margin-bottom: 6px;
            }


            .dashboard-bar-track {
                width: 100%;
                max-width: 38px;
                height: 145px;
                display: flex;
                align-items: flex-end;
                background: #f5f4fa;
                border-radius: 9px 9px 5px 5px;
                overflow: hidden;
            }


            .dashboard-bar-fill {
                width: 100%;
                min-height: 3px;
                border-radius: 8px 8px 4px 4px;
                background:
                    linear-gradient(
                        180deg,
                        #7c6df5,
                        #5d4be2
                    );
                transition:
                    height 0.7s
                    cubic-bezier(.2,.8,.2,1);
            }


            .dashboard-bar-label {
                width: 55px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                text-align: center;
                margin-top: 8px;
                color: #8b90a2;
                font-size: 9px;
            }


            /* =========================================
               HISTORY TABLE
               ========================================= */

            .dashboard-history-panel {
                margin-bottom: 20px;
            }


            .dashboard-table-wrapper {
                overflow-x: auto;
            }


            .dashboard-history-table {
                width: 100%;
                border-collapse: collapse;
                min-width: 650px;
            }


            .dashboard-history-table th {
                text-align: left;
                padding:
                    12px 13px;
                background: #fafafe;
                border-bottom:
                    1px solid #ededf4;
                color: #8a90a2;
                font-size: 10px;
                text-transform: uppercase;
                letter-spacing: 0.55px;
            }


            .dashboard-history-table td {
                padding:
                    14px 13px;
                border-bottom:
                    1px solid #f0f0f5;
                color: #444a5f;
                font-size: 12px;
            }


            .dashboard-history-table tr:last-child td {
                border-bottom: none;
            }


            .dashboard-file-cell {
                max-width: 220px;
                font-weight: 700;
                color: #292d40 !important;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }


            .dashboard-type-pill {
                display: inline-flex;
                align-items: center;
                padding:
                    5px 9px;
                border-radius: 20px;
                background: #f3f3f8;
                color: #73798c;
                font-size: 10px;
                font-weight: 800;
            }


            .dashboard-result-pill {
                display: inline-flex;
                align-items: center;
                padding:
                    5px 10px;
                border-radius: 20px;
                font-size: 10px;
                font-weight: 850;
            }


            .dashboard-result-pill.real {
                background: #e8faf2;
                color: #159966;
            }


            .dashboard-result-pill.fake {
                background: #fff0f0;
                color: #d45151;
            }

            .dashboard-result-pill.review {
                background: #fff5e5;
                color: #b87514;
            }


            .dashboard-confidence {
                font-weight: 800;
                color: #555b70;
            }

            .dashboard-trend-chart {
                display: grid;
                height: 150px;
                grid-template-columns: repeat(7, minmax(0, 1fr));
                align-items: end;
                gap: 10px;
                padding-top: 12px;
            }

            .dashboard-trend-day {
                display: flex;
                height: 100%;
                min-width: 0;
                flex-direction: column;
                align-items: center;
                justify-content: flex-end;
                gap: 7px;
            }

            .dashboard-trend-day span {
                display: flex;
                width: 100%;
                min-height: 3px;
                align-items: flex-end;
                justify-content: center;
                border-radius: 6px 6px 2px 2px;
                background: #f0effa;
            }

            .dashboard-trend-day i {
                display: block;
                width: 100%;
                border-radius: inherit;
                background: linear-gradient(180deg, #7666f2, #5b65d8);
            }

            .dashboard-trend-day small {
                color: #858ba0;
                font-size: 9px;
            }

            .dashboard-score-bins {
                display: grid;
                min-height: 150px;
                grid-template-columns: repeat(5, minmax(0, 1fr));
                align-items: end;
                gap: 10px;
                padding-top: 12px;
            }

            .dashboard-score-bin {
                display: flex;
                height: 100%;
                flex-direction: column;
                align-items: center;
                justify-content: flex-end;
                gap: 6px;
                color: #858ba0;
                font-size: 9px;
            }

            .dashboard-score-bin-track {
                display: flex;
                width: 100%;
                height: 105px;
                align-items: flex-end;
                overflow: hidden;
                border-radius: 6px 6px 2px 2px;
                background: #f3f4f9;
            }

            .dashboard-score-bin-track i {
                display: block;
                width: 100%;
                min-height: 2px;
                border-radius: 6px 6px 0 0;
                background: #7365e8;
            }

            .dashboard-score-bin strong {
                color: #4a5067;
                font-size: 10px;
            }

            .dashboard-frame-distribution {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 8px;
            }

            .dashboard-frame-distribution div,
            .dashboard-investigation-stats span {
                display: flex;
                flex-direction: column;
                gap: 5px;
                padding: 10px;
                border: 1px solid #ececf3;
                border-radius: 9px;
                color: #858ba0;
                font-size: 10px;
            }

            .dashboard-frame-distribution strong,
            .dashboard-investigation-stats strong {
                color: #292e43;
                font-size: 14px;
            }

            .dashboard-frame-bar {
                display: flex;
                height: 9px;
                margin: 14px 0;
                overflow: hidden;
                border-radius: 10px;
                background: #f0f1f5;
            }

            .dashboard-frame-bar span {
                display: block;
                height: 100%;
                transition: width 0.4s ease;
            }

            #dashboardRealFrameBar {
                background: #18a874;
            }

            #dashboardFakeFrameBar {
                background: #df5b5b;
            }

            #dashboardSuspiciousFrameBar {
                background: #e1a33a;
            }

            .dashboard-investigation-stats {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 8px;
            }


            .dashboard-empty-history {
                text-align: center;
                padding: 48px 20px;
                color: #999eaf;
            }


            .dashboard-empty-history-icon {
                font-size: 30px;
                margin-bottom: 10px;
            }


            .dashboard-empty-history strong {
                display: block;
                color: #656b7e;
                margin-bottom: 4px;
            }


            /* =========================================
               RESPONSIVE
               ========================================= */

            @media (max-width: 1050px) {

                .dashboard-stat-grid {
                    grid-template-columns:
                        repeat(2, minmax(0, 1fr));
                }

                .dashboard-chart-grid {
                    grid-template-columns: 1fr;
                }

            }


            @media (max-width: 650px) {

                .dashboard-heading {
                    align-items: flex-start;
                    flex-direction: column;
                }

                .dashboard-heading-actions {
                    width: 100%;
                }

                .dashboard-refresh-button,
                .dashboard-clear-button {
                    flex: 1;
                }

                .dashboard-stat-grid {
                    grid-template-columns: 1fr;
                }

                .dashboard-donut-area {
                    flex-direction: column;
                    gap: 20px;
                }

                .dashboard-bars {
                    gap: 7px;
                }

            }

        `;


        document.head.appendChild(
            style
        );
    }


    /* =========================================================
       DASHBOARD HTML
       ========================================================= */

    function createDashboard() {

        if (!dashboardSection) {
            console.warn(
                "DeepGuard dashboard section was not found."
            );
            return;
        }


        if (
            document.getElementById(
                "deepguardDashboard"
            )
        ) {
            return;
        }


        injectDashboardStyles();


        const dashboard =
            document.createElement(
                "div"
            );

        dashboard.id =
            "deepguardDashboard";

        dashboard.className =
            "deepguard-dashboard";


        dashboard.innerHTML = `

            <div class="dashboard-heading">

                <div>
                    <h2>Detection Analytics</h2>

                    <p>
                        Monitor media analyses,
                        classification trends,
                        and recent detection activity.
                    </p>
                </div>


                <div class="dashboard-heading-actions">

                    <button
                        type="button"
                        class="dashboard-refresh-button"
                        id="dashboardRefreshButton"
                    >
                        ↻ Refresh
                    </button>

                    <button
                        type="button"
                        class="dashboard-clear-button"
                        id="dashboardClearButton"
                    >
                        Clear History
                    </button>

                </div>

            </div>


            <!-- STAT CARDS -->

            <div class="dashboard-stat-grid">

                <div class="dashboard-stat-card">

                    <div class="dashboard-stat-icon">
                        ◈
                    </div>

                    <div class="dashboard-stat-label">
                        Total analyses
                    </div>

                    <div
                        class="dashboard-stat-value"
                        id="dashboardTotal"
                    >
                        0
                    </div>

                    <div class="dashboard-stat-subtext">
                        Successful detections
                    </div>

                </div>


                <div class="dashboard-stat-card real">

                    <div class="dashboard-stat-icon">
                        ✓
                    </div>

                    <div class="dashboard-stat-label">
                        Real detections
                    </div>

                    <div
                        class="dashboard-stat-value"
                        id="dashboardReal"
                    >
                        0
                    </div>

                    <div class="dashboard-stat-subtext">
                        Classified as authentic
                    </div>

                </div>


                <div class="dashboard-stat-card fake">

                    <div class="dashboard-stat-icon">
                        !
                    </div>

                    <div class="dashboard-stat-label">
                        Fake detections
                    </div>

                    <div
                        class="dashboard-stat-value"
                        id="dashboardFake"
                    >
                        0
                    </div>

                    <div class="dashboard-stat-subtext">
                        Classified as manipulated
                    </div>

                </div>


                <div class="dashboard-stat-card review">

                    <div class="dashboard-stat-icon">
                        ?
                    </div>

                    <div class="dashboard-stat-label">
                        Needs review
                    </div>

                    <div
                        class="dashboard-stat-value"
                        id="dashboardReview"
                    >
                        0
                    </div>

                    <div class="dashboard-stat-subtext">
                        Inconclusive assessments
                    </div>

                </div>


                <div class="dashboard-stat-card confidence">

                    <div class="dashboard-stat-icon">
                        ◎
                    </div>

                    <div class="dashboard-stat-label">
                        Avg. Fake-class score
                    </div>

                    <div
                        class="dashboard-stat-value"
                        id="dashboardAverage"
                    >
                        0%
                    </div>

                    <div class="dashboard-stat-subtext">
                        Across recorded analyses
                    </div>

                </div>

            </div>


            <!-- CHARTS -->

            <div class="dashboard-chart-grid">

                <div class="dashboard-panel">

                    <div class="dashboard-panel-header">

                        <div>
                            <h3 class="dashboard-panel-title">
                                Detection distribution
                            </h3>

                            <p class="dashboard-panel-description">
                                Real versus fake classifications
                            </p>
                        </div>

                    </div>


                    <div class="dashboard-donut-area">

                        <div
                            class="dashboard-donut"
                            id="dashboardDonut"
                        >

                            <div class="dashboard-donut-center">

                                <span
                                    class="dashboard-donut-number"
                                    id="dashboardDonutNumber"
                                >
                                    0
                                </span>

                                <span
                                    class="dashboard-donut-label"
                                >
                                    analyses
                                </span>

                            </div>

                        </div>


                        <div class="dashboard-legend">

                            <div class="dashboard-legend-item">

                                <span
                                    class="dashboard-legend-dot real"
                                ></span>

                                <div
                                    class="dashboard-legend-text"
                                >

                                    <span
                                        class="dashboard-legend-name"
                                    >
                                        Real
                                    </span>

                                    <span
                                        class="dashboard-legend-value"
                                        id="dashboardRealLegend"
                                    >
                                        0
                                    </span>

                                </div>

                            </div>

                            <div class="dashboard-legend-item">

                                <span class="dashboard-legend-dot review"></span>

                                <div class="dashboard-legend-text">
                                    <span class="dashboard-legend-name">Needs Review</span>
                                    <span class="dashboard-legend-value" id="dashboardReviewLegend">0</span>
                                </div>

                            </div>

                            <div class="dashboard-legend-item">

                                <span
                                    class="dashboard-legend-dot fake"
                                ></span>

                                <div
                                    class="dashboard-legend-text"
                                >

                                    <span
                                        class="dashboard-legend-name"
                                    >
                                        Fake
                                    </span>

                                    <span
                                        class="dashboard-legend-value"
                                        id="dashboardFakeLegend"
                                    >
                                        0
                                    </span>

                                </div>

                            </div>

                        </div>

                    </div>

                </div>

                <div class="dashboard-panel">
                    <div class="dashboard-panel-header">
                        <div>
                            <h3 class="dashboard-panel-title">Detection score distribution</h3>
                            <p class="dashboard-panel-description">Bounded Fake-class model scores, not probabilities</p>
                        </div>
                    </div>
                    <div class="dashboard-score-bins" id="dashboardScoreDistribution"></div>
                </div>


                <div class="dashboard-panel">

                    <div class="dashboard-panel-header">

                        <div>
                            <h3 class="dashboard-panel-title">Recent model scores</h3>

                            <p class="dashboard-panel-description">
                                Bounded SVM scores, not calibrated probabilities
                            </p>
                        </div>

                    </div>

                    <div
                        class="dashboard-bars"
                        id="dashboardBars"
                    >
                    </div>

                </div>


                <div class="dashboard-panel">
                    <div class="dashboard-panel-header">
                        <div>
                            <h3 class="dashboard-panel-title">Recent analysis trend</h3>
                            <p class="dashboard-panel-description">Successful analyses over the last seven days</p>
                        </div>
                    </div>
                    <div class="dashboard-trend-chart" id="dashboardTrendChart"></div>
                </div>


                <div class="dashboard-panel">
                    <div class="dashboard-panel-header">
                        <div>
                            <h3 class="dashboard-panel-title">Video frame assessments</h3>
                            <p class="dashboard-panel-description">Counts across video analyses in local history</p>
                        </div>
                    </div>
                    <div class="dashboard-frame-distribution">
                        <div><span>Real-like</span><strong id="dashboardRealFrames">0</strong></div>
                        <div><span>Fake-like</span><strong id="dashboardFakeFrames">0</strong></div>
                        <div><span>Suspicious</span><strong id="dashboardSuspiciousFrames">0</strong></div>
                    </div>
                    <div class="dashboard-frame-bar" aria-label="Video frame assessment distribution">
                        <span id="dashboardRealFrameBar"></span>
                        <span id="dashboardFakeFrameBar"></span>
                        <span id="dashboardSuspiciousFrameBar"></span>
                    </div>
                    <div class="dashboard-investigation-stats">
                        <span>Total frames analyzed <strong id="dashboardFrameTotal">0</strong></span>
                        <span>Faces detected <strong id="dashboardFaceTotal">0</strong></span>
                        <span>Average face-detection rate <strong id="dashboardFaceRate">Not available</strong></span>
                    </div>
                </div>

            </div>


            <!-- HISTORY -->

            <div class="dashboard-panel dashboard-history-panel">

                <div class="dashboard-panel-header">

                    <div>

                        <h3 class="dashboard-panel-title">
                            Detection history
                        </h3>

                        <p class="dashboard-panel-description">
                            Your latest successful media analyses
                        </p>

                    </div>

                    <div
                        class="dashboard-panel-description"
                        id="dashboardHistoryCount"
                    >
                        0 records
                    </div>

                </div>


                <div
                    class="dashboard-table-wrapper"
                    id="dashboardTableContainer"
                >
                </div>

            </div>

        `;


        /*
         * Add the dashboard after
         * the existing dashboard content.
         */
        dashboardSection.appendChild(
            dashboard
        );


        const refreshButton =
            document.getElementById(
                "dashboardRefreshButton"
            );

        const clearButton =
            document.getElementById(
                "dashboardClearButton"
            );


        if (refreshButton) {

            refreshButton.addEventListener(
                "click",
                updateDashboard
            );

        }


        if (clearButton) {

            clearButton.addEventListener(
                "click",
                () => {

                    const history =
                        getDetectionHistory();

                    if (
                        history.length === 0
                    ) {
                        return;
                    }


                    const confirmed =
                        window.confirm(
                            "Clear all DeepGuard AI detection history?"
                        );


                    if (!confirmed) {
                        return;
                    }


                    localStorage.removeItem(
                        DASHBOARD_HISTORY_KEY
                    );


                    updateDashboard();

                }
            );

        }


        updateDashboard();
    }


    function updateAnalysisTrend(history) {

        const chart =
            document.getElementById("dashboardTrendChart");

        if (!chart) {
            return;
        }


        const today =
            new Date();

        today.setHours(0, 0, 0, 0);

        const trendDays =
            Array.from(
                { length: 7 },
                (_, index) => {

                    const date =
                        new Date(today);

                    date.setDate(
                        today.getDate() - (6 - index)
                    );

                    return {
                        date,
                        key: date.toDateString(),
                        count: 0,
                        label: date.toLocaleDateString(
                            undefined,
                            { weekday: "short" }
                        )
                    };

                }
            );


        history.forEach(
            (item) => {

                const timestamp =
                    new Date(item.timestamp);

                if (!Number.isFinite(timestamp.getTime())) {
                    return;
                }

                const day =
                    trendDays.find(
                        (entry) =>
                            entry.key === timestamp.toDateString()
                    );

                if (day) {
                    day.count += 1;
                }

            }
        );


        const maximum =
            Math.max(
                1,
                ...trendDays.map((day) => day.count)
            );

        chart.innerHTML =
            trendDays.map(
                (day) => {

                    const height =
                        day.count
                            ? Math.max(8, (day.count / maximum) * 100)
                            : 0;

                    return `
                        <div class="dashboard-trend-day" title="${day.count} analyses">
                            <strong>${day.count}</strong>
                            <span><i style="height:${height}%"></i></span>
                            <small>${day.label}</small>
                        </div>
                    `;

                }
            ).join("");

    }


    /* =========================================================
       UPDATE DASHBOARD
       ========================================================= */

    function updateDashboard() {

        if (
            !document.getElementById(
                "deepguardDashboard"
            )
        ) {
            return;
        }


        const history =
            getDetectionHistory();


        const total =
            history.length;


        const realCount =
            history.filter(
                item =>
                    String(
                        item.label
                    ).toLowerCase() ===
                    "real"
            ).length;


        const fakeCount =
            history.filter(
                item =>
                    String(
                        item.label
                    ).toLowerCase() ===
                    "fake"
            ).length;

        const reviewCount =
            history.filter(
                item =>
                    String(item.label).toLowerCase() ===
                    "needs review"
            ).length;

        const modelScoreValues =
            history
                .map(
                    (item) => {

                        if (
                            item.model_score != null &&
                            Number.isFinite(Number(item.model_score))
                        ) {
                            return Number(item.model_score);
                        }

                        if (
                            item.confidence == null ||
                            !Number.isFinite(Number(item.confidence))
                        ) {
                            return null;
                        }

                        const confidence =
                            Number(item.confidence);

                        return String(item.label).toLowerCase() === "fake"
                            ? confidence
                            : 100 - confidence;

                    }
                )
                .filter(
                    value =>
                        value != null &&
                        Number.isFinite(value)
                );


        const averageModelScore =
            modelScoreValues.length
                ? modelScoreValues.reduce(
                    (
                        totalValue,
                        value
                    ) =>
                        totalValue +
                        value,
                    0
                ) /
                modelScoreValues.length
                : 0;

        const totalFrames =
            history.reduce(
                (sum, item) => sum + (Number(item.analyzed_frames) || 0),
                0
            );

        const totalFaces =
            history.reduce(
                (sum, item) => sum + (Number(item.face_detected_frames) || 0),
                0
            );

        const totalRealFrames =
            history.reduce(
                (sum, item) => sum + (
                    Number(item.real_like_frames ?? item.real_frames) || 0
                ),
                0
            );

        const totalFakeFrames =
            history.reduce(
                (sum, item) => sum + (
                    Number(item.fake_like_frames ?? item.fake_frames) || 0
                ),
                0
            );

        const totalSuspiciousFrames =
            history.reduce(
                (sum, item) => sum + (Number(item.suspicious_frames) || 0),
                0
            );

        const faceDetectionRate =
            totalFrames > 0
                ? (totalFaces / totalFrames) * 100
                : null;


        /* =========================================
           STAT CARDS
           ========================================= */

        const totalElement =
            document.getElementById(
                "dashboardTotal"
            );

        const realElement =
            document.getElementById(
                "dashboardReal"
            );

        const fakeElement =
            document.getElementById(
                "dashboardFake"
            );

        const averageElement =
            document.getElementById(
                "dashboardAverage"
            );

        const reviewElement =
            document.getElementById("dashboardReview");


        if (totalElement) {
            totalElement.textContent =
                total;
        }

        if (realElement) {
            realElement.textContent =
                realCount;
        }

        if (fakeElement) {
            fakeElement.textContent =
                fakeCount;
        }

        if (reviewElement) {
            reviewElement.textContent =
                reviewCount;
        }

        if (averageElement) {
            averageElement.textContent =
                modelScoreValues.length
                    ? averageModelScore.toFixed(
                    1
                    ) + "%"
                    : "Not available";
        }


        /* =========================================
           DONUT CHART
           ========================================= */

        const donut =
            document.getElementById(
                "dashboardDonut"
            );

        const donutNumber =
            document.getElementById(
                "dashboardDonutNumber"
            );

        const realLegend =
            document.getElementById(
                "dashboardRealLegend"
            );

        const fakeLegend =
            document.getElementById(
                "dashboardFakeLegend"
            );

        const reviewLegend =
            document.getElementById(
                "dashboardReviewLegend"
            );


        if (donutNumber) {
            donutNumber.textContent =
                total;
        }

        if (realLegend) {
            realLegend.textContent =
                realCount;
        }

        if (fakeLegend) {
            fakeLegend.textContent =
                fakeCount;
        }

        if (reviewLegend) {
            reviewLegend.textContent =
                reviewCount;
        }


        if (donut) {

            if (total === 0) {

                donut.style.background =
                    "#f1f0f7";

            } else {

                const realPercentage =
                    (
                        realCount /
                        total
                    ) *
                    100;

                const fakePercentage =
                    (
                        fakeCount /
                        total
                    ) *
                    100;

                const reviewPercentage =
                    (
                        reviewCount /
                        total
                    ) *
                    100;

                const reviewStart =
                    realPercentage + fakePercentage;

                donut.style.background =
                    `conic-gradient(
                        #18a874 0% ${realPercentage}%,
                        #df5b5b ${realPercentage}% ${reviewStart}%,
                        #e1a33a ${reviewStart}% ${reviewStart + reviewPercentage}%
                    )`;

            }

        }

        const frameStats = {
            dashboardRealFrames: totalRealFrames,
            dashboardFakeFrames: totalFakeFrames,
            dashboardSuspiciousFrames: totalSuspiciousFrames,
            dashboardFrameTotal: totalFrames,
            dashboardFaceTotal: totalFaces,
            dashboardFaceRate: faceDetectionRate == null
                ? "Not available"
                : `${faceDetectionRate.toFixed(1)}%`
        };

        Object.entries(frameStats).forEach(
            ([id, value]) => {

                const element =
                    document.getElementById(id);

                if (element) {
                    element.textContent =
                        String(value);
                }

            }
        );

        updateAnalysisTrend(history);

        const scoredFrameTotal =
            totalRealFrames + totalFakeFrames + totalSuspiciousFrames;

        const frameBarWidths = {
            dashboardRealFrameBar:
                scoredFrameTotal
                ? (totalRealFrames / scoredFrameTotal) * 100
                    : 0,
            dashboardFakeFrameBar:
                scoredFrameTotal
                ? (totalFakeFrames / scoredFrameTotal) * 100
                    : 0,
            dashboardSuspiciousFrameBar:
                scoredFrameTotal
                    ? (totalSuspiciousFrames / scoredFrameTotal) * 100
                    : 0
        };

        Object.entries(frameBarWidths).forEach(
            ([id, width]) => {

                const element =
                    document.getElementById(id);

                if (element) {
                    element.style.width =
                        `${width}%`;
                }

            }
        );

        const scoreDistribution =
            document.getElementById("dashboardScoreDistribution");

        if (scoreDistribution) {

            const scoreBins = [
                { label: "0–20", count: 0 },
                { label: "21–40", count: 0 },
                { label: "41–60", count: 0 },
                { label: "61–80", count: 0 },
                { label: "81–100", count: 0 }
            ];

            modelScoreValues.forEach(
                (score) => {

                    const index =
                        Math.min(
                            4,
                            Math.floor(score / 20)
                        );

                    scoreBins[index].count += 1;

                }
            );

            const maxBin =
                Math.max(
                    1,
                    ...scoreBins.map((bin) => bin.count)
                );

            scoreDistribution.innerHTML =
                scoreBins.map(
                    (bin) => {

                        const height =
                            bin.count
                                ? Math.max(4, (bin.count / maxBin) * 100)
                                : 0;

                        return `
                            <div class="dashboard-score-bin">
                                <strong>${bin.count}</strong>
                                <span class="dashboard-score-bin-track">
                                    <i style="height:${height}%"></i>
                                </span>
                                <small>${bin.label}</small>
                            </div>
                        `;

                    }
                ).join("");

        }


        /* =========================================
           RECENT MODEL SCORE BAR CHART
           ========================================= */

        const barsContainer =
            document.getElementById(
                "dashboardBars"
            );


        if (barsContainer) {

            const recent =
                history
                    .slice(0, 8)
                    .reverse();


            if (recent.length === 0) {

                barsContainer.innerHTML = `
                    <div
                        style="
                            width:100%;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            color:#9a9faf;
                            font-size:12px;
                        "
                    >
                        Analyze media to populate the chart.
                    </div>
                `;

            } else {

                barsContainer.innerHTML =
                    recent
                        .map(
                            (
                                item,
                                index
                            ) => {

                                const score =
                                    Math.max(
                                        0,
                                        Math.min(
                                            100,
                                            item.model_score != null
                                                ? Number(item.model_score) || 0
                                                : String(item.label).toLowerCase() === "fake"
                                                    ? Number(item.confidence) || 0
                                                    : 100 - (Number(item.confidence) || 0)
                                        )
                                    );


                                const shortName =
                                    String(
                                        item.filename ||
                                        "File"
                                    )
                                        .replace(
                                            /\.[^/.]+$/,
                                            ""
                                        );


                                const displayName =
                                    shortName.length >
                                    8
                                        ? shortName.slice(
                                            0,
                                            7
                                        ) + "…"
                                        : shortName;


                                return `
                                    <div
                                        class="dashboard-bar-item"
                                        title="${escapeDashboardHTML(
                                            item.filename
                                        )}"
                                    >

                                        <div
                                            class="dashboard-bar-value"
                                        >
                                            ${score.toFixed(0)}%
                                        </div>

                                        <div
                                            class="dashboard-bar-track"
                                        >

                                            <div
                                                class="dashboard-bar-fill"
                                                style="
                                                    height:${score}%;
                                                "
                                            ></div>

                                        </div>

                                        <div
                                            class="dashboard-bar-label"
                                        >
                                            ${escapeDashboardHTML(
                                                displayName
                                            )}
                                        </div>

                                    </div>
                                `;

                            }
                        )
                        .join("");

            }

        }


        /* =========================================
           HISTORY TABLE
           ========================================= */

        const tableContainer =
            document.getElementById(
                "dashboardTableContainer"
            );


        const historyCount =
            document.getElementById(
                "dashboardHistoryCount"
            );


        if (historyCount) {

            historyCount.textContent =
                `${total} ${
                    total === 1
                        ? "record"
                        : "records"
                }`;

        }


        if (!tableContainer) {
            return;
        }


        if (history.length === 0) {

            tableContainer.innerHTML = `

                <div class="dashboard-empty-history">

                    <div
                        class="dashboard-empty-history-icon"
                    >
                        ◫
                    </div>

                    <strong>
                        No detection history yet
                    </strong>

                    <span>
                        Upload an image or video
                        in Media Analyzer to begin.
                    </span>

                </div>

            `;

            return;
        }


        const recentHistory =
            history.slice(0, 10);


        tableContainer.innerHTML = `

            <table
                class="dashboard-history-table"
            >

                <thead>

                    <tr>

                        <th>
                            Media
                        </th>

                        <th>
                            Type
                        </th>

                        <th>
                            Result
                        </th>

                        <th>
                            Model Score
                        </th>

                        <th>
                            Analyzed
                        </th>

                    </tr>

                </thead>


                <tbody>

                    ${recentHistory
                        .map(
                            item => {

                                const label =
                                    String(
                                        item.label ||
                                        "Unknown"
                                    );


                                const labelClass =
                                    label.toLowerCase() ===
                                    "fake"
                                        ? "fake"
                                        : label.toLowerCase() ===
                                            "needs review"
                                            ? "review"
                                            : "real";


                                const modelScore =
                                    item.model_score != null
                                        ? Number(item.model_score)
                                        : item.confidence != null
                                            ? label.toLowerCase() === "fake"
                                                ? Number(item.confidence)
                                                : 100 - Number(item.confidence)
                                            : null;


                                return `

                                    <tr>

                                        <td
                                            class="dashboard-file-cell"
                                            title="${escapeDashboardHTML(
                                                item.filename
                                            )}"
                                        >
                                            ${escapeDashboardHTML(
                                                item.filename
                                            )}
                                        </td>


                                        <td>

                                            <span
                                                class="dashboard-type-pill"
                                            >
                                                ${escapeDashboardHTML(
                                                    item.media_type
                                                )}
                                            </span>

                                        </td>


                                        <td>

                                            <span
                                                class="dashboard-result-pill ${labelClass}"
                                            >
                                                ${
                                                    labelClass ===
                                                    "fake"
                                                        ? "!"
                                                        : labelClass === "review"
                                                            ? "?"
                                                        : "✓"
                                                }

                                                &nbsp;

                                                ${escapeDashboardHTML(
                                                    label
                                                )}
                                            </span>

                                        </td>


                                        <td
                                            class="dashboard-confidence"
                                        >
                                            ${
                                                modelScore != null &&
                                                Number.isFinite(modelScore)
                                                    ? modelScore.toFixed(
                                                        2
                                                    )
                                                    : "—"
                                            }${modelScore != null ? "%" : ""}
                                        </td>


                                        <td>
                                            ${formatDashboardDate(
                                                item.timestamp
                                            )}
                                        </td>

                                    </tr>

                                `;

                            }
                        )
                        .join("")}

                </tbody>

            </table>

        `;

    }


    /* =========================================================
       INITIALIZE DASHBOARD
       ========================================================= */

    createDashboard();

});