import cv2
from argparse import ArgumentParser
from insightface.app import FaceAnalysis
from util import create_temp_names, draw_faces, read_image
from fr import UserFaceSet


def main():
    parser = ArgumentParser()
    parser.add_argument("--embedding_dir", type=str,
                        default="registered_faces",
                        help="Directory containing face embeddings")
    parser.add_argument("-o", "--output", type=str,
                        default="test.jpg",
                        help="Output directory")
    parser.add_argument("--top_n", type=int,
                        default=None,
                        help="Number of top matches to print")
    parser.add_argument("-u", "--user", type=str, required=True,
                        help="User name to check")
    parser.add_argument("image_path", type=str,
                        help="Path to the image file")
    args = parser.parse_args()

    image_path = args.image_path
    user_name = args.user
    embedding_dir = args.embedding_dir
    output_file = args.output
    top_n = args.top_n

    app = FaceAnalysis()
    app.prepare(ctx_id=-1)
    # app.prepare(ctx_id=0)  # GPU

    # 顔特徴ベクトルのロード
    users_faceset = UserFaceSet.from_dir(embedding_dir, target_user=user_name)
    if len(users_faceset.users_faces) == 0:
        print(f"No registered faces for {user_name}.")
        return

    img = read_image(image_path)

    # 顔の検出
    faces = app.get(img)
    if len(faces) == 0:
        print("No faces are found.")
        return
    else:
        print(f"Found {len(faces)} faces.")

    match_results = []
    for face in faces:
        match_result = users_faceset.most_match_face(face)
        match_results.append(match_result)

    # 類似度の高い順にソート
    match_results = sorted(match_results,
                           key=lambda x: x.similarity,
                           reverse=True)

    tmp_names = create_temp_names(len(match_results))
    similarities = [m.similarity for m in match_results]

    if top_n is not None:
        faces = faces[:top_n]
        tmp_names = tmp_names[:top_n]
        similarities = similarities[:top_n]

    for name, sim in zip(tmp_names, similarities):
        print(f"{name}: {sim:.3f}")

    img = draw_faces(img, faces, tmp_names)

    # 検出結果を保存
    cv2.imwrite(output_file, img)


if __name__ == "__main__":
    main()
