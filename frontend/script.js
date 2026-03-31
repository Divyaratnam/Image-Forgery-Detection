// ---------------- SECTION SWITCHING ----------------
function showSection(sectionId) {
    document.querySelectorAll("section").forEach(sec => {
        sec.classList.remove("active");
    });
    document.getElementById(sectionId).classList.add("active");
}

// ---------------- IMAGE PREVIEW ----------------
document.getElementById("imageInput").addEventListener("change", function () {
    const file = this.files[0];
    const preview = document.getElementById("preview");

    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";

        document.getElementById("analysisBox").style.display = "none";
        document.getElementById("detailsBox").style.display = "none";
    }
});

// ---------------- ANALYZE IMAGE ----------------
function analyzeImage() {

    const input = document.getElementById("imageInput");
    const resultText = document.getElementById("finalResult");
    const confidenceText = document.getElementById("confidenceText");

    if (input.files.length === 0) {
        alert("Please upload an image first!");
        return;
    }

    const file = input.files[0];
    const formData = new FormData();
    formData.append("image", file);

    document.getElementById("analysisBox").style.display = "block";

    resultText.innerText = "⏳ Analyzing image...";
    confidenceText.innerText = "";

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
    console.log("Backend response:", data);

    const imageName = file.name;
    const imageSize = (file.size / 1024).toFixed(2) + " KB";
    const imageFormat = file.type.split("/")[1].toUpperCase();

    const imgObj = new Image();
    imgObj.src = URL.createObjectURL(file);

    imgObj.onload = function () {

        const resolution = imgObj.width + " × " + imgObj.height;

        let resultLabel;
        let confidence = data.confidence + "%";

    if (data.prediction === "FAKE") {

       resultLabel = "FORGED";

       resultText.innerText = "🛑 Result: Forged Image";
       confidenceText.innerText = "Confidence: " + confidence;

       document.getElementById("forgedBar").style.width = confidence;
       document.getElementById("genuineBar").style.width = (100 - data.confidence) + "%";

       document.getElementById("forgedPercent").innerText = confidence;
       document.getElementById("genuinePercent").innerText = (100 - data.confidence).toFixed(1) + "%";
    }

    else if (data.prediction === "REAL") {

       resultLabel = "GENUINE";

       resultText.innerText = "✅ Result: Genuine Image";
       confidenceText.innerText = "Confidence: " + confidence;

       document.getElementById("genuineBar").style.width = confidence;
       document.getElementById("forgedBar").style.width = (100 - data.confidence) + "%";

       document.getElementById("genuinePercent").innerText = confidence;
      document.getElementById("forgedPercent").innerText = (100 - data.confidence).toFixed(1) + "%";
    } 
        // ✅ SHOW ANALYSIS BOX
        document.getElementById("analysisBox").style.display = "block";

        // ✅ DETAILS BOX
        document.getElementById("detailsBox").style.display = "block";
        document.getElementById("imgName").innerText = imageName;
        document.getElementById("imgFormat").innerText = imageFormat;
        document.getElementById("imgResolution").innerText = resolution;
        document.getElementById("imgSize").innerText = imageSize;
        document.getElementById("imgResult").innerText = resultLabel;
        document.getElementById("imgConfidence").innerText = confidence;

        // ✅ REPORT TEXT (REAL CONFIDENCE)
        window.reportText =
`IMAGE FORGERY DETECTION REPORT
-----------------------------

Image Information
-----------------
Image Name      : ${imageName}
Image Format    : ${imageFormat}
Resolution      : ${resolution}
File Size       : ${imageSize}

Analysis Result
---------------
Result          : ${resultLabel}
Confidence      : ${confidence}
Status          : Completed

Model Details
-------------
Architecture    : GAN + Swin Transformer + EfficientNet
Decision Method : Decision-Level Fusion
Inference Device: CPU

Remarks
-------
Prediction generated using hybrid deep learning fusion.
`;

        document.getElementById("downloadBtn").disabled = false;
        document.getElementById("downloadMsg").innerText =
            "Analysis complete. You can download the report.";
    };
})

    .catch(err => {
        console.error(err);
        alert("❌ Backend connection failed!");
    });
}

// ---------------- DOWNLOAD REPORT ----------------
function downloadReport() {

    if (!window.reportText) {
        alert("Please analyze an image first!");
        return;
    }

    const blob = new Blob([window.reportText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "image_forgery_detection_report.txt";
    a.click();

    URL.revokeObjectURL(url);
}
