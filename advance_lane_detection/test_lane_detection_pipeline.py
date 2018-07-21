from lane_detection_pipeline import LaneDetectionPipeline

def test_prepare_pipeline():
    pipeline = LaneDetectionPipeline()
    pipeline.prepare_pipeline("../camera_cal", (9,6))
