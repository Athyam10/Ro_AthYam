// Ensure the displayed <img id="originalDisplay"> is drawn to the canvas
function ensureImageDrawn(canvas, ctx, callback) {
  const img = document.getElementById('originalDisplay');

  function doDraw() {
    // Use the image's natural size for the canvas
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    ctx.drawImage(img, 0, 0);
    if (typeof callback === 'function') callback();
  }

  if (img.complete && img.naturalWidth) {
    doDraw();
  } else {
    img.addEventListener('load', doDraw, { once: true });
  }
}

// Open popup
function openPopup() {
  document.getElementById("popup").style.display = "flex";

  const canvas = document.getElementById("convertedCanvas");
  const ctx = canvas.getContext("2d");

  // Draw the original image into the canvas (when loaded)
  ensureImageDrawn(canvas, ctx);
}

// Close popup
function closePopup() {

  document.getElementById("popup").style.display = "none";
}

// Run brightness conversion
function runConversion() {
  const canvas = document.getElementById("convertedCanvas");
  const ctx = canvas.getContext("2d");

  // Get brightness value
  let brightness = parseInt(document.getElementById("brightnessInput").value) || 0;

  // Ensure the original image is drawn first, then run conversion
  ensureImageDrawn(canvas, ctx, () => {
    // Get pixels
    let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let data = imageData.data;

    // Loop through pixels
    for (let i = 0; i < data.length; i += 4) {
      let r = data[i] / 255;
      let g = data[i + 1] / 255;
      let b = data[i + 2] / 255;

      // RGB -> HSV
      let max = Math.max(r, g, b);
      let min = Math.min(r, g, b);
      let diff = max - min;

      let h = 0, s = 0, v = max;

      s = max === 0 ? 0 : diff / max;

      // Hue
      if (diff === 0) {
        h = 0;
      } else if (max === r) {
        h = (60 * ((g - b) / diff) + 360) % 360;
      } else if (max === g) {
        h = (60 * ((b - r) / diff) + 120) % 360;
      } else {
        h = (60 * ((r - g) / diff) + 240) % 360;
      }

      // Change brightness
      v += brightness / 100;

      // Clamp
      v = Math.max(0, Math.min(1, v));

      // HSV -> RGB
      let c = v * s;
      let x = c * (1 - Math.abs((h / 60) % 2 - 1));
      let m = v - c;

      let r1, g1, b1;

      if (h < 60) {
        r1 = c; g1 = x; b1 = 0;
      } else if (h < 120) {
        r1 = x; g1 = c; b1 = 0;
      } else if (h < 180) {
        r1 = 0; g1 = c; b1 = x;
      } else if (h < 240) {
        r1 = 0; g1 = x; b1 = c;
      } else if (h < 300) {
        r1 = x; g1 = 0; b1 = c;
      } else {
        r1 = c; g1 = 0; b1 = x;
      }

      // Convert back to RGB
      data[i]     = Math.round((r1 + m) * 255);
      data[i + 1] = Math.round((g1 + m) * 255);
      data[i + 2] = Math.round((b1 + m) * 255);
    }

    // Put edited image back
    ctx.putImageData(imageData, 0, 0);
  });
}