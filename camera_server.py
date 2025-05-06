from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_cors import cross_origin
from picamera2 import Picamera2
import threading, time
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

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

@app.route("/start", methods=["POST"])
@cross_origin() 
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

@app.route("/stop", methods=["POST"])
@cross_origin() 
def stop():
    global picam2, running
    running = False
    time.sleep(0.3)
    picam2.stop()
    picam2.close()
    return jsonify(status="stopped")

@app.route('/video_feed')
@cross_origin() 
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def smooth_array(arr, window_size=9):
    if window_size < 2:
        return arr
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(arr, kernel, mode='same')

def process_image(crop):
    import pandas as pd
    import traceback
    from scipy.signal import find_peaks

    try:
        print("[DEBUG] Starting image processing")

        # calibration constants
        a = 4.483335
        b = 253.462921

        # grayscale & sum columns
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        print("[DEBUG] Grayscale conversion complete")

        intensity = np.sum(gray, axis=0).astype(np.float32)
        print(f"[DEBUG] Intensity array shape: {intensity.shape}")

        # smooth
        smooth = smooth_array(intensity, window_size=9)
        print(f"[DEBUG] Smoothed intensity shape: {smooth.shape}")

        # wavelengths axis
        pixels = np.arange(len(smooth))
        wavelengths = a * pixels + b

        # prepare raw-data text for 300–800 nm
        data_pts = [(w, v) for w, v in zip(wavelengths, smooth) if 300 <= w <= 800]
        data_pts.sort(key=lambda x: x[0])
        lines = [" nm      %", "------------"] + [f"{w:.1f}\t{v:.1f}" for w, v in data_pts]
        data_str = "\n".join(lines)
        print("[DEBUG] Raw data string prepared")

        # dataframe for peak detection and plotting
        df = pd.DataFrame({'nm': wavelengths, 'percentage': smooth})
        df = df[(df["nm"] >= 350) & (df["nm"] <= 750)].copy()
        print(f"[DEBUG] Trimmed DataFrame shape: {df.shape}")

        peaks, _ = find_peaks(df["percentage"], height=60)
        print(f"[DEBUG] Detected {len(peaks)} peaks")

        peak_positions = df.iloc[peaks].reset_index(drop=True)

        def filter_peaks(peak_df, threshold=25):
            peak_df = peak_df.sort_values(by='nm')
            filtered = []
            for _, row in peak_df.iterrows():
                if not filtered or (row['nm'] - filtered[-1]['nm']) > threshold:
                    filtered.append(row)
                else:
                    if row['percentage'] > filtered[-1]['percentage']:
                        filtered[-1] = row
            return pd.DataFrame(filtered)

        filtered_peaks = filter_peaks(peak_positions)
        print(f"[DEBUG] Filtered to {len(filtered_peaks)} peaks")

        # plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["nm"], df["percentage"], color='black', linewidth=1.2)
        ax.scatter(filtered_peaks["nm"], filtered_peaks["percentage"], color='red', s=40, zorder=5)

        ax.set_xlim(350, 750)
        ax.set_xticks(np.arange(350, 751, 25))
        ax.set_xlabel("Wavelength (nm)", fontsize=12, weight='bold')
        ax.set_ylabel("Intensity", fontsize=12, weight='bold')
        ax.tick_params(width=2, length=6)
        ax.grid(True, linestyle="--", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        if not filtered_peaks.empty:
            legend_text = "Peaks: " + ", ".join(
                f"{row['nm']:.1f} nm ({row['percentage']:.0f})"
                for _, row in filtered_peaks.iterrows()
            )
            ax.legend([legend_text], loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=10)

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)

        # encode plot
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200)
        buf.seek(0)
        spectrum_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        print("[DEBUG] Spectrum plot encoding complete")

        # encode crop
        _, imgbuf = cv2.imencode('.png', crop)
        cropped_b64 = base64.b64encode(imgbuf).decode('utf-8')
        print("[DEBUG] Cropped image encoding complete")

        return cropped_b64, spectrum_b64, data_str

    except Exception as e:
        print("[ERROR] Exception in process_image:")
        traceback.print_exc()
        raise e

@app.route("/capture", methods=["POST"])
@cross_origin() 
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

    height, width = crop.shape[:2]
    squeezed_crop = cv2.resize(crop, (width, int(height * 0.5)), interpolation=cv2.INTER_AREA)

    crop = cv2.resize(squeezed_crop, (width, height), interpolation=cv2.INTER_LINEAR)

    try:
        cr, sp, dt = process_image(crop)
        return jsonify(cropped=cr, spectrum=sp, data=dt)
    except Exception as e:
        # if something goes wrong in processing, send JSON error
        return jsonify(error=str(e)), 500
    
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/<path:any_path>", methods=["OPTIONS"])
def handle_preflight(any_path):
    return ("", 204)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
