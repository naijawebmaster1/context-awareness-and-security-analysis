import cv2
import numpy as np
import torch
from scipy.signal import convolve2d
from skimage.feature import local_binary_pattern
from networkb import MLPLivenessClassifier

MODE = "nn_gradients"
MODEL_PATH = f"models/liveness_{MODE}.pth"
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def single_scale_retinex(img, sigma=50):
    img_float = np.float32(img) + 1.0
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
    retinex = np.log10(img_float) - np.log10(blurred)
    return np.uint8(cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX))


def extract_lpq_histogram(img, win_size=7, freq=1.0):
    img = np.float64(img)
    r = (win_size - 1) / 2
    x = np.arange(-r, r + 1)[np.newaxis]
    w0 = np.ones_like(x)
    w1 = np.exp(-2 * np.pi * 1j * x * freq / win_size)
    w2 = np.conj(w1)
    q1, q2, q3, q4 = w0.T * w1, w1.T * w0, w1.T * w1, w1.T * w2
    filters = [np.real(q1), np.imag(q1), np.real(q2), np.imag(q2),
               np.real(q3), np.imag(q3), np.real(q4), np.imag(q4)]
    lpq_img = np.zeros(img.shape, dtype=np.uint8)
    for i, f in enumerate(filters):
        response = convolve2d(img, f, mode='same', boundary='symm')
        lpq_img += (response > 0).astype(np.uint8) << i
    hist, _ = np.histogram(lpq_img.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def extract_features(face_bgr, mode):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    retinex = single_scale_retinex(gray)

    if mode == "nn_gradients":
        resized = cv2.resize(retinex, (64, 64))
        gx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX).flatten()

    elif mode == "nn_raw_gray":
        return (cv2.resize(retinex, (64, 64)).astype(np.float32) / 255.0).flatten()

    elif mode == "nn_spatial_lbp":
        lbp = local_binary_pattern(cv2.resize(retinex, (64, 64)), P=16, R=2, method="uniform")
        return (lbp.astype(np.float32) / 18.0).flatten()

    elif mode == "nn_high_freq":
        lap = cv2.Laplacian(cv2.resize(retinex, (64, 64)), cv2.CV_64F)
        return cv2.normalize(lap, None, 0, 1, cv2.NORM_MINMAX).flatten()

    elif mode == "nn_lpq":
        return extract_lpq_histogram(cv2.resize(retinex, (64, 64)), win_size=7)

    return None


def predict_frame(classifier, face_bgr, mode):
    feat = extract_features(face_bgr, mode)
    if feat is None:
        return None, None
    feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(classifier.device)
    classifier.model.eval()
    with torch.no_grad():
        logit = classifier.model(feat_tensor)
        prob = torch.sigmoid(logit).item()
    label = "LIVE" if prob >= 0.5 else "SPOOF"
    return label, prob


def main():
    print(f"Loading model: {MODEL_PATH}")
    classifier = MLPLivenessClassifier.load_model(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray_full, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            label, prob = predict_frame(classifier, face_crop, MODE)

            if label is None:
                continue

            color = (0, 200, 0) if label == "LIVE" else (0, 0, 220)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            confidence = prob if label == "LIVE" else 1 - prob
            text = f"{label}  {confidence*100:.1f}%"
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Liveness Detection — Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
