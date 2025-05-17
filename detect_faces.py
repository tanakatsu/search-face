import cv2
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from util import create_temp_names, draw_faces, read_image


def save_face_embeddings(faces: list[Face], names: list[str], output_dir: str) -> None:
    assert len(faces) == len(names), "Number of faces and names must match"

    for face, name in zip(faces, names):
        embedding = face.embedding
        output_path = Path(output_dir) / f"{name}.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            np.save(f, embedding)


def save_face_images(img: np.ndarray, faces: list[Face], names: list[str], output_dir: str) -> None:
    assert len(faces) == len(names), "Number of faces and names must match"

    for face, name in zip(faces, names):
        output_path = Path(output_dir) / f"{name}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        left, top, right, bottom = face.bbox.astype(int)
        face_img = img[top:bottom, left:right]

        cv2.imwrite(output_path, face_img)


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="detected")
    parser.add_argument("--save_face_image", action="store_true",
                        help="Save face images")
    parser.add_argument("img_path", type=str)
    args = parser.parse_args()

    img_path = args.img_path
    output_dir = args.output_dir
    save_face_image_flag = args.save_face_image
    import pdb; pdb.set_trace()

    img = read_image(img_path)

    app = FaceAnalysis()
    app.prepare(ctx_id=-1)
    # app.prepare(ctx_id=0)  # When you use GPU, set ctx_id=0

    # 顔検出
    faces = app.get(img)

    if len(faces) == 0:
        print("No faces detected.")
    else:
        print(f"{len(faces)} faces detected.")

        # 顔の埋め込みベクトルを保存する
        names = create_temp_names(len(faces))
        save_face_embeddings(faces, names, output_dir)

        if save_face_image_flag:
            # 顔切り出し画像を保存する
            save_face_images(img, faces, names, output_dir)

        # 顔の検出結果を表示
        img = draw_faces(img, faces, names)

        # 検出結果画像を保存
        cv2.imwrite(Path(output_dir) / "result.jpg", img)


if __name__ == "__main__":
    main()
