import cv2
import numpy as np
import shutil
from collections import Counter
from argparse import ArgumentParser
from pathlib import Path
from insightface.app import FaceAnalysis
from util import draw_faces, read_image
from fr import UserFaceSet, MatchResult


FILE_TYPES = ("jpg", "jpeg", "JPG", "JPEG", "png", "PNG",
              "heic", "HEIC", "heif", "HEIF")


def get_photos(photo_dir: str) -> list[Path]:
    filelist = []
    for ext in FILE_TYPES:
        filelist.extend(Path(photo_dir).glob(f"**/*.{ext}"))
    return list(filelist)


def save_photo(img: np.ndarray, match_result: MatchResult,
               image_path: Path, photo_dir: str, output_dir: str,
               no_result_copy: bool = False, flat_output: bool = False,
               suffix: str = ""):
    user_name = match_result.name
    if flat_output:
        filename = image_path.name
        if suffix:
            filename = filename.replace(".", f"{suffix}.")  # example.jpg -> example(1).jpg
    else:
        filename = str(image_path).replace(f"{photo_dir}/", "")

    # Original image
    output_path = Path(output_dir) / user_name / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(image_path, output_path)
    print(f"Saved {output_path}")

    # Image for confirmation
    if no_result_copy:
        return
    confirm_output_path = Path(output_dir) / "_confirm" / user_name / filename
    confirm_output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(confirm_output_path.with_suffix(".jpg"), img)


def find_duplicates(filelist: list[Path]) -> dict[str, int]:
    filenames = [f.name for f in filelist]
    dup_files = {name: 0 for name, cnt in Counter(filenames).items() if cnt > 1}
    return dup_files


def get_suffix_num(file_path: Path, dup_files: dict[Path, int]) -> str:
    filename = file_path.name
    if filename not in dup_files:
        return ""

    cnt = dup_files[filename]
    if cnt == 0:  # first one
        return ""

    # second or later
    return f"({cnt})"


def main():
    parser = ArgumentParser()
    parser.add_argument("--embedding_dir", type=str,
                        default="registered_faces",
                        help="Directory containing face embeddings")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("-o", "--output_dir", type=str,
                        default="output",
                        help="Output directory")
    parser.add_argument("--no_result_copy",
                        action="store_true",
                        help="Do not copy the detected result image")
    parser.add_argument("--flat_output",
                        action="store_true",
                        help="Flatten the output directory structure")
    parser.add_argument("photo_dir", type=str,
                        help="Path to the image directory")
    args = parser.parse_args()

    photo_dir = args.photo_dir
    embedding_dir = args.embedding_dir
    output_dir = args.output_dir
    threshold = args.threshold
    no_result_copy = args.no_result_copy
    flat_output = args.flat_output

    app = FaceAnalysis()
    app.prepare(ctx_id=-1)
    # app.prepare(ctx_id=0)  # GPU

    # 顔特徴ベクトルのロード
    users_faceset = UserFaceSet.from_dir(embedding_dir)
    if len(users_faceset.users_faces) == 0:
        print("No registered faces.")
        return

    filelist = get_photos(photo_dir)
    if len(filelist) == 0:
        print("No pictures are found.")
        return
    print(f"Found {len(filelist)} pictures.")

    duplicated_files = find_duplicates(filelist)

    for file_path in filelist:
        print(f"Processing {file_path}...")
        img = read_image(file_path)

        # 顔の検出
        faces = app.get(img)

        match_results = []
        for face in faces:
            match_result = users_faceset.most_match_face(face)
            match_results.append(match_result)

        matched_faces = []
        for match_result in match_results:
            if match_result.similarity >= threshold:
                print(f"Matched face: {match_result.name} ({match_result.similarity})")
                matched_faces.append(match_result)

        if len(matched_faces) == 0:
            print("No matched face.")
            continue

        print(f"Number of matched faces: {len(matched_faces)}")
        faces, names = zip(*[[m.face, m.name] for m in matched_faces])
        img = draw_faces(img, faces, names)

        # 検出結果を保存
        for match_result in matched_faces:
            if flat_output:
                suffix = get_suffix_num(file_path, duplicated_files)
                if suffix:
                    duplicated_files[file_path] += 1
            else:
                suffix = ""
            save_photo(img, match_result, file_path, photo_dir, output_dir,
                       no_result_copy=no_result_copy,
                       flat_output=flat_output,
                       suffix=suffix)


if __name__ == "__main__":
    main()
