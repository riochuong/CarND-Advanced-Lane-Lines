from lane_detection_pipeline import LaneDetectionPipeline


if __name__ == "__main__":
    pipeline = LaneDetectionPipeline()
    pipeline.prepare_pipeline("../camera_cal", (9,6))
    pipeline.run("../project_video.mp4")
