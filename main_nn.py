import os
import json
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm
from networkb import MLPLivenessClassifier
from scipy.signal import convolve2d

def extract_lpq_histogram(img, win_size=3, freq=1.0):
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

def single_scale_retinex(img, sigma=50):
    img_float = np.float32(img) + 1.0
    blurred = cv2.GaussianBlur(img_float, (0, 0), sigma)
    retinex = np.log10(img_float) - np.log10(blurred)
    return np.uint8(cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX))

def process_cropped_image(image_path, mode):
    img = cv2.imread(image_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retinex = single_scale_retinex(gray)

    if mode == 'nn_gradients':
        resized = cv2.resize(retinex, (64, 64))
        gx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX).flatten()

    elif mode == 'nn_raw_gray':
        return (cv2.resize(retinex, (64, 64)).astype(np.float32) / 255.0).flatten()

    elif mode == 'nn_spatial_lbp':
        lbp = local_binary_pattern(cv2.resize(retinex, (64, 64)), P=16, R=2, method='uniform')
        return (lbp.astype(np.float32) / 18.0).flatten()

    elif mode == 'nn_high_freq':
        lap = cv2.Laplacian(cv2.resize(retinex, (64, 64)), cv2.CV_64F)
        return cv2.normalize(lap, None, 0, 1, cv2.NORM_MINMAX).flatten()
    
    elif mode == "nn_lpq":
        return extract_lpq_histogram(cv2.resize(retinex, (64, 64)), win_size=7)
    
class CASIAEvaluator:
    def __init__(self, dataset_path, mode):
        self.dataset_path = dataset_path
        self.mode = mode
        self.cache_dir = "extracted_features"
        os.makedirs(self.cache_dir, exist_ok=True)
        self.classifier = MLPLivenessClassifier()

    def _load_features(self, split):
        cache_path = os.path.join(self.cache_dir, f"{split}_{self.mode}.npz")
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            return data['X'], data['y']

        X, y = [], []
        split_dir = os.path.join(self.dataset_path, split)
        for cat, label in {'live': 1, 'spoof': 0}.items():
            path = os.path.join(split_dir, cat)
            for f in tqdm(os.listdir(path), desc=f"Extracting {split} {cat}"):
                feat = process_cropped_image(os.path.join(path, f), self.mode)
                if feat is not None: X.append(feat); y.append(label)
        
        X, y = np.array(X), np.array(y)
        np.savez_compressed(cache_path, X=X, y=y)
        return X, y

    def evaluate(self):
        X_train, y_train = self._load_features('train')
        self.classifier.train(X_train, y_train)
        self.classifier.save_model(f"models/liveness_{self.mode}.pth")

        X_test, y_test = self._load_features('test')
        preds = self.classifier.predict(X_test)

        apcer = np.sum((preds == 1) & (y_test == 0)) / np.sum(y_test == 0)
        bpcer = np.sum((preds == 0) & (y_test == 1)) / np.sum(y_test == 1)
        acer  = (apcer + bpcer) * 50
        print(f"Mode: {self.mode}\nAPCER: {apcer*100:.2f}%\nBPCER: {bpcer*100:.2f}%\nACER: {acer:.2f}%")

        results_path = "results.json"
        results = json.loads(open(results_path).read()) if os.path.exists(results_path) else {}
        results[self.mode] = {
            "APCER": round(apcer * 100, 2),
            "BPCER": round(bpcer * 100, 2),
            "ACER":  round(acer, 2),
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    #Set your task
    CHOSEN_MODE = 'nn_lpq' 
    dataset_path = "./casia-fasd"
    evaluator = CASIAEvaluator(dataset_path, CHOSEN_MODE)
    evaluator.evaluate()