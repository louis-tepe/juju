import cv2
import numpy as np


def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crops the black borders of an image based on a tolerance level."""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img


def ben_graham_processing(image: np.ndarray, sigmaX: int = 10) -> np.ndarray:
    """
    Applies Ben Graham's preprocessing method:
    Formula: image = 4 * image - 4 * GaussianBlur(image) + 128
    """
    image = crop_image_from_gray(image)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


def preprocess_image(path: str, size: int = 256, sigmaX: int = 10) -> np.ndarray:
    """Reads, crops, resizes, and applies Ben Graham processing."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (size, size))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    return image


def load_and_preprocess_image(path: str, size: int = 256, sigmaX: int = 10) -> np.ndarray:
    """
    Full pipeline: Read -> Preprocess (Graham) -> Resize -> Normalize (0-1).
    Used for Inference and Validation.
    """
    image = preprocess_image(path, size, sigmaX)
    return image.astype(np.float32) / 255.0
