from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from picamera2 import Picamera2
import threading, time
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)
CORS(app)

picam2 = None
frame = None
lock = threading.Lock()
running = False

def capture_frames():
    global frame, running
    while running:
        img = picam2.capture_array()
        with lock:
            frame = img.copy()
        time.sleep(0.03)

def gen_frames():
    global running, frame
    while True:
        if running:
            with lock:
                if frame is None:
                    continue
                ret, buf = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() +
                   b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/start', methods=['POST'])
def start():
    global picam2, running
    if running:
        return jsonify(status="running")
    try:
        picam2 = Picamera2()
        cfg = picam2.create_preview_configuration(main={"size": (640,480)})
        picam2.configure(cfg)
        picam2.start()
        running = True
        threading.Thread(target=capture_frames, daemon=True).start()
        return jsonify(status="started")
    except Exception as e:
        return jsonify(status="error", message=str(e)), 500

@app.route('/stop', methods=['POST'])
def stop():
    global picam2, running
    running = False
    time.sleep(0.3)
    picam2.stop()
    picam2.close()
    return jsonify(status="stopped")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def smooth_array(arr, window_size=9):
    if window_size < 2:
        return arr
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(arr, kernel, mode='same')

def process_image(crop):
    # calibration constants
    a = 3.553578
    b = 304.797351
    # a = -4.361064
    # b = 810.141268

    # grayscale & sum columns
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    intensity = np.sum(gray, axis=0).astype(np.float32)

    # smooth
    smooth = smooth_array(intensity, window_size=9)

    # wavelengths axis
    pixels      = np.arange(len(smooth))
    wavelengths = a * pixels + b

    # find peaks in 400–700 nm
    peaks = [
        i for i in range(1, len(smooth)-1)
        if smooth[i] > smooth[i-1]
        and smooth[i] > smooth[i+1]
        and 400 <= (a*i + b) <= 700
    ]

    # pick highest in each 25 nm window
    selected = []
    for i in sorted(peaks, key=lambda i: smooth[i], reverse=True):
        wl = a*i + b
        if all(abs(wl - (a*j + b)) >= 25 for j in selected):
            selected.append(i)
    selected.sort(key=lambda i: a*i + b)

    # prepare raw-data text sorted by ascending λ
    data_pts = [(w, v) for w, v in zip(wavelengths, smooth) if 400 <= w <= 700]
    data_pts.sort(key=lambda x: x[0])
    lines = [" nm      %", "------------"] + [f"{w:.1f}\t{v:.1f}" for w,v in data_pts]
    data_str = "\n".join(lines)

    # plotting
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(wavelengths, smooth, color='black', linewidth=1.2)
    ax.scatter([a*i + b for i in selected],
               [smooth[i] for i in selected],
               color='red', s=40, zorder=5)

    # X axis: 350→750 nm, ticks every 25 nm
    ax.set_xlim(350, 750)
    ax.set_xticks(np.arange(350, 751, 25))
    ax.set_xlabel("Wavelength (nm)")

    # Y axis: 0 → 10% above max intensity
    y_max = smooth.max() * 1.10
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Intensity")

    ax.set_title("")
    ax.grid(True, linestyle="--", alpha=0.5)

    # legend below
    legend_text = "Peaks: " + ", ".join(
        f"{(a*i+b):.1f} nm ({smooth[i]:.0f})" for i in selected
    )
    ax.legend([legend_text],
              loc='upper center',
              bbox_to_anchor=(0.5, -0.15),
              fontsize=10)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    # encode plot & crop
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200)
    buf.seek(0)
    spectrum_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    _, imgbuf = cv2.imencode('.png', crop)
    cropped_b64 = base64.b64encode(imgbuf).decode('utf-8')

    return cropped_b64, spectrum_b64, data_str

@app.route('/capture', methods=['POST'])
def capture():
    global frame

    data = request.get_json(force=True)
    x1, y1 = int(data['x1']), int(data['y1'])
    x2, y2 = int(data['x2']), int(data['y2'])

    # grab the latest frame under the lock
    with lock:
        if frame is None:
            return jsonify(error="No frame available"), 503
        # snap = frame.copy()
        snap = cv2.flip(frame.copy(), 1)

    # crop bounds safely
    h, w = snap.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    crop = snap[y1:y2, x1:x2]

    try:
        cr, sp, dt = process_image(crop)
        return jsonify(cropped=cr, spectrum=sp, data=dt)
    except Exception as e:
        # if something goes wrong in processing, send JSON error
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
