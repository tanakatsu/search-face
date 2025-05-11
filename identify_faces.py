import cv2
import numpy as np
import shutil
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from util import draw_faces


def get_photos(photo_dir):
    filelist = Path(photo_dir).glob("**/*.jpg")
    return list(filelist)


def save_photo(img, match_result, image_path, photo_dir, output_dir):
    user_name = match_result.name
    filename = str(image_path).replace(f"{photo_dir}/", "")

    # Original image
    output_path = Path(output_dir) / user_name / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(image_path, output_path)

    # Image for confirmation
    confirm_output_path = Path(output_dir) / "_confirm" / user_name / filename
    confirm_output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(confirm_output_path, img)
    print(f"Saved {output_path}")


# 類似度の算出のための関数（コサイン類似度）
def cos_sim(feat1, feat2):
    feat1_norm = np.linalg.norm(feat1)
    feat2_norm = np.linalg.norm(feat2)
    return np.dot(feat1, feat2) / (feat1_norm * feat2_norm)


@dataclass
class MatchResult:
    name: str
    similarity: float
    face: Face


class UserFace:
    def __init__(self, name):
        self.name = name
        self.embeddings = []

    def match_face(self, face):
        similarities = [cos_sim(emb, face.embedding) for emb in self.embeddings]
        return MatchResult(self.name, max(similarities), face)

    def load(self, user_embedding_dir):
        embedding_files = Path(user_embedding_dir).glob("*.npy")
        for embedding_file in embedding_files:
            embedding = np.load(embedding_file)
            self.embeddings.append(embedding)


class UserFaceSet():
    def __init__(self, users_faces: list[UserFace]):
        self.users_faces = users_faces

    def most_match_face(self, face):
        match_results = []
        for user_face in self.users_faces:
            result = user_face.match_face(face)
            match_results.append(result)

        most_match_face = sorted(match_results,
                                 key=lambda x: x.similarity,
                                 reverse=True)[0]
        return most_match_face

    @classmethod
    def from_dir(cls, embedding_dir):
        users_faces = []
        for user_dir in Path(embedding_dir).iterdir():
            if user_dir.is_file():
                continue

            user_name = user_dir.name
            user_face = UserFace(user_name)
            user_face.load(user_dir)

            users_faces.append(user_face)

        return cls(users_faces)


def main():
    parser = ArgumentParser()
    parser.add_argument("--embedding_dir", type=str,
                        default="registered_faces",
                        help="Directory containing face embeddings")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("-o", "--output_dir", type=str,
                        default="output",
                        help="Output directory")
    parser.add_argument("photo_dir", type=str,
                        help="Path to the image directory")
    args = parser.parse_args()

    photo_dir = args.photo_dir
    embedding_dir = args.embedding_dir
    output_dir = args.output_dir
    threshold = args.threshold

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

    for file_path in filelist:
        print(f"Processing {file_path}...")
        img = cv2.imread(file_path)

        # 顔の検出
        faces = app.get(img)

        matched_faces = []
        for face in faces:
            match_result = users_faceset.most_match_face(face)
            if match_result.similarity >= threshold:
                print(f"Matched face: {match_result.name} ({match_result.similarity})")
                matched_faces.append(match_result)

        if len(matched_faces) == 0:
            print("No matched face.")
            break

        print(f"Number of matched faces: {len(matched_faces)}")
        faces, names = zip(*[[m.face, m.name] for m in matched_faces])
        img = draw_faces(img, faces, names)

        # 検出結果を保存
        for match_result in matched_faces:
            save_photo(img, match_result, file_path, photo_dir, output_dir)


if __name__ == "__main__":
    main()
