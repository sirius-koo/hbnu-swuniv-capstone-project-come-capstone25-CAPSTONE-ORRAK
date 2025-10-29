import gradio as gr
import os
from analysis import analyze_video
import shutil

os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

def process_video(video_file):
    if video_file is None:
        return None

    upload_path = os.path.join("uploads", os.path.basename(video_file))
    shutil.copy(video_file, upload_path)

    base_name = os.path.basename(upload_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    result_path = os.path.join("results", f"{file_name_without_ext}.mp4")

    analyze_video(upload_path, result_path)

    return result_path

demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(
        label="영상 업로드"
        ),
    outputs=gr.Video(
        label="분석 결과"
        ),
    title="초보 클라이머를 위한 자세 추정 기반 분석 도구",
    description=(
        "🟡 **무게중심점** | 🟢 **안정(기저면 내부)** | 🔴 **불안정(기저면 외부)**  \n\n"
        "무게중심점이 초록색 기저면 안에 위치하면 **안정 상태**,  \n"
        "벗어나면 해당 면이 **빨간색으로 전환**되어 불안정 상태로 표시됩니다."
    ),
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch(
        share=True,
        # auth=[("admin", "1234")]
        )

