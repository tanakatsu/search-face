import numpy as np
from dataclasses import dataclass
from insightface.app.common import Face
from pathlib import Path


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
    def from_dir(cls, embedding_dir, target_user=None):
        users_faces = []
        for user_dir in Path(embedding_dir).iterdir():
            if user_dir.is_file():
                continue

            user_name = user_dir.name
            if target_user is not None and target_user != user_name:
                continue

            user_face = UserFace(user_name)
            user_face.load(user_dir)

            users_faces.append(user_face)

        return cls(users_faces)
