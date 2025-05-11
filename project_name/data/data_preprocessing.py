import os
import cv2
import numpy as np
import itertools

INPUT_DIRECTORY = "Data"
OUTPUT_DIRECTORY = "Data_Preprocessed"

translate_params = {
    "dx": {"min": -10, "max": 10, "step": 5},
    "dy": {"min": -10, "max": 10, "step": 5},
}
rotate_params = {
    "angle": {"min": -30, "max": 30, "step": 15},
    "scale": {"min": 0.5, "max": 1.5, "step": 0.5},
}
skew_params = {
    "dx": {"min": -0.2, "max": 0.2, "step": 0.1},
    "dy": {"min": -0.2, "max": 0.2, "step": 0.1},
}
elastic_params = {
    "alpha": {"min": 1.0, "max": 5.0, "step": 2.0},
    "sigma": {"min": 4.0, "max": 6.0, "step": 2.0},
}
flip_axes = [0, 1, -1]  # 0=vertical, 1=horizontal, -1=both axes

rgb_shift_params = {
    "r": {"min": -50, "max": 50, "step": 25},
    "g": {"min": -50, "max": 50, "step": 25},
    "b": {"min": -50, "max": 50, "step": 25},
}
hsv_params = {
    "h": {"min": -10, "max": 10, "step": 5},
    "s": {"min": 0.8, "max": 1.2, "step": 0.2},
    "v": {"min": 0.8, "max": 1.2, "step": 0.2},
}
channel_shuffle_params = {
    "orders": list(
        itertools.permutations([0, 1, 2], 3)
    )  # all 6 possible channel orders
}
clahe_params = {
    "clipLimit": [2.0, 4.0],
    "tileGridSize": [(8, 8), (16, 16)],
}
random_contrast_params = {
    "alpha": {"min": 0.5, "max": 1.5, "step": 0.5},
}
random_gamma_params = {
    "gamma": {"min": 0.7, "max": 1.5, "step": 0.4},
}
random_brightness_params = {
    "beta": {"min": -50, "max": 50, "step": 25},
}


def translate_image(image, dx, dy):
    rows, columns = image.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        image, M, (columns, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )


def rotate_image(image, angle, scale=1.0):
    rows, columns = image.shape[:2]
    center = (columns / 2, rows / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(
        image, M, (columns, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )


def skew_image(image, dx=0.0, dy=0.0):
    rows, columns = image.shape[:2]
    M = np.float32([[1, dx, 0], [dy, 1, 0]])
    return cv2.warpAffine(
        image, M, (columns, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )


def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape[:2]
    dx = random_state.uniform(-1, 1, size=shape).astype(np.float32)
    dy = random_state.uniform(-1, 1, size=shape).astype(np.float32)
    dx = cv2.GaussianBlur(dx, (0, 0), sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigma)
    dx *= alpha
    dy *= alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def flip_image(image, axis):
    return cv2.flip(image, axis)


def rgb_shift(image, r, g, b):
    # split, add offsets, clip, merge
    b_ch, g_ch, r_ch = cv2.split(image)
    r_ch = np.clip(r_ch + r, 0, 255).astype(np.uint8)
    g_ch = np.clip(g_ch + g, 0, 255).astype(np.uint8)
    b_ch = np.clip(b_ch + b, 0, 255).astype(np.uint8)
    return cv2.merge((b_ch, g_ch, r_ch))


def adjust_hsv(image, h_shift, s_scale, v_scale):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + h_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * s_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * v_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def channel_shuffle(image, order):
    ch = cv2.split(image)
    shuffled = [ch[i] for i in order]
    return cv2.merge(shuffled)


def apply_clahe(image, clipLimit, tileGridSize):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def random_contrast(image, alpha):
    # alpha >1 increases contrast; <1 decreases
    image = image.astype(np.float32)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    return np.clip((image - mean) * alpha + mean, 0, 255).astype(np.uint8)


def random_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype(
        "uint8"
    )
    return cv2.LUT(image, table)


def random_brightness(image, beta):
    return np.clip(image.astype(np.int16) + beta, 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    for index, file_name in enumerate(sorted(os.listdir(INPUT_DIRECTORY)), start=1):
        if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            continue
        image_path = os.path.join(INPUT_DIRECTORY, file_name)
        image = cv2.imread(image_path)

        for dx in np.arange(
            translate_params["dx"]["min"],
            translate_params["dx"]["max"] + translate_params["dx"]["step"],
            translate_params["dx"]["step"],
        ):
            for dy in np.arange(
                translate_params["dy"]["min"],
                translate_params["dy"]["max"] + translate_params["dy"]["step"],
                translate_params["dy"]["step"],
            ):
                transformed = translate_image(image, dx, dy)
                output_name = f"Image_{index}_Translated_{dx}_{dy}.png"
                cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, output_name), transformed)

        for angle in np.arange(
            rotate_params["angle"]["min"],
            rotate_params["angle"]["max"] + rotate_params["angle"]["step"],
            rotate_params["angle"]["step"],
        ):
            for scale in np.arange(
                rotate_params["scale"]["min"],
                rotate_params["scale"]["max"] + rotate_params["scale"]["step"],
                rotate_params["scale"]["step"],
            ):
                transformed = rotate_image(image, angle, scale)
                output_name = f"Image_{index}_Rotated_{angle}_{scale}.png"
                cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, output_name), transformed)

        for dx in np.arange(
            skew_params["dx"]["min"],
            skew_params["dx"]["max"] + skew_params["dx"]["step"],
            skew_params["dx"]["step"],
        ):
            for dy in np.arange(
                skew_params["dy"]["min"],
                skew_params["dy"]["max"] + skew_params["dy"]["step"],
                skew_params["dy"]["step"],
            ):
                transformed = skew_image(image, dx, dy)
                output_name = f"Image_{index}_Skewed_{dx}_{dy}.png"
                cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, output_name), transformed)

        for alpha in np.arange(
            elastic_params["alpha"]["min"],
            elastic_params["alpha"]["max"] + elastic_params["alpha"]["step"],
            elastic_params["alpha"]["step"],
        ):
            for sigma in np.arange(
                elastic_params["sigma"]["min"],
                elastic_params["sigma"]["max"] + elastic_params["sigma"]["step"],
                elastic_params["sigma"]["step"],
            ):
                transformed = elastic_transform(image, alpha, sigma)
                output_name = f"Image_{index}_Elastic_{alpha}_{sigma}.png"
                cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, output_name), transformed)

        for axis in flip_axes:
            transformed = flip_image(image, axis)
            output_name = f"Image_{index}_Flipped_{axis}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, output_name), transformed)

        for r in np.arange(
            rgb_shift_params["r"]["min"],
            rgb_shift_params["r"]["max"] + rgb_shift_params["r"]["step"],
            rgb_shift_params["r"]["step"],
        ):
            for g in np.arange(
                rgb_shift_params["g"]["min"],
                rgb_shift_params["g"]["max"] + rgb_shift_params["g"]["step"],
                rgb_shift_params["g"]["step"],
            ):
                for b in np.arange(
                    rgb_shift_params["b"]["min"],
                    rgb_shift_params["b"]["max"] + rgb_shift_params["b"]["step"],
                    rgb_shift_params["b"]["step"],
                ):
                    transformed = rgb_shift(image, r, g, b)
                    output_name = f"Image_{index}_RGB_{r}_{g}_{b}.png"
                    cv2.imwrite(
                        os.path.join(OUTPUT_DIRECTORY, output_name), transformed
                    )
        for h in np.arange(
            hsv_params["h"]["min"],
            hsv_params["h"]["max"] + hsv_params["h"]["step"],
            hsv_params["h"]["step"],
        ):
            for s in np.arange(
                hsv_params["s"]["min"],
                hsv_params["s"]["max"] + hsv_params["s"]["step"],
                hsv_params["s"]["step"],
            ):
                for v in np.arange(
                    hsv_params["v"]["min"],
                    hsv_params["v"]["max"] + hsv_params["v"]["step"],
                    hsv_params["v"]["step"],
                ):
                    transformed = adjust_hsv(image, h, s, v)
                    output_name = f"Image_{index}_HSV_{h}_{s:.1f}_{v:.1f}.png"
                    cv2.imwrite(
                        os.path.join(OUTPUT_DIRECTORY, output_name), transformed
                    )

        for order in channel_shuffle_params["orders"]:
            transformed = channel_shuffle(image, order)
            output_name = f"Image_{index}_CHS_{order[0]}{order[1]}{order[2]}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, output_name), transformed)

        for clip in clahe_params["clipLimit"]:
            for tgs in clahe_params["tileGridSize"]:
                transformed = apply_clahe(image, clip, tgs)
                output_name = f"Image_{index}_CLAHE_{clip}_{tgs[0]}x{tgs[1]}.png"
                cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, output_name), transformed)

        for alpha in np.arange(
            random_contrast_params["alpha"]["min"],
            random_contrast_params["alpha"]["max"]
            + random_contrast_params["alpha"]["step"],
            random_contrast_params["alpha"]["step"],
        ):
            transformed = random_contrast(image, alpha)
            output_name = f"Image_{index}_CON_{alpha:.1f}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, output_name), transformed)

        for gamma in np.arange(
            random_gamma_params["gamma"]["min"],
            random_gamma_params["gamma"]["max"] + random_gamma_params["gamma"]["step"],
            random_gamma_params["gamma"]["step"],
        ):
            transformed = random_gamma(image, gamma)
            output_name = f"Image_{index}_GAM_{gamma:.2f}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, output_name), transformed)

        for beta in np.arange(
            random_brightness_params["beta"]["min"],
            random_brightness_params["beta"]["max"]
            + random_brightness_params["beta"]["step"],
            random_brightness_params["beta"]["step"],
        ):

            transformed = random_brightness(image, beta)
            output_name = f"Image_{index}_BRT_{beta}.png"
            cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, output_name), transformed)

        print(f"Processed {file_name} ({index})")


if __name__ == "__main__":
    main()
