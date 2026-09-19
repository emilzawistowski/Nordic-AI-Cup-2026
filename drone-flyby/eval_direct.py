import sys
import argparse
from typing import Dict, List, Optional, Tuple
from dtos import DroneFlybyPredictRequestDto, DroneFlybyPredictResponseDto
from utils import frame_numbers, load_frame, DEFAULT_SCENE, global_bbox_to_source
from local_evaluator import Camera, CameraRejection, render_view, build_request, score, Statistics
from example import predict

def run_direct(scene: str = DEFAULT_SCENE):
    frames = frame_numbers(scene)
    statistics = Statistics(frames_total=len(frames))
    predictions: Dict[int, List[dict]] = {}
    camera = Camera()
    feedback: Optional[dict] = None

    for frame_index, frame in enumerate(frames):
        image = load_frame(frame, scene)
        encoded_image = render_view(image, camera)
        payload = build_request(frame, frame_index, camera, encoded_image, feedback)
        
        request_dto = DroneFlybyPredictRequestDto.model_validate(payload)
        response_dto = predict(request_dto)
        
        predictions[frame] = [
            {
                'object_id': annotation.object_id,
                'bbox': global_bbox_to_source(
                    annotation.bbox,
                    payload['original_width'],
                    payload['original_height'],
                ),
                'confidence': float(annotation.confidence),
            }
            for annotation in response_dto.annotations
        ]
        
        # update camera feedback and position
        requested = response_dto.requested_view
        if requested is not None:
            try:
                camera.apply(
                    requested.resolution_level,
                    requested.center_x,
                    requested.center_y,
                )
                statistics.commands_applied += 1
                feedback = None
            except CameraRejection as exc:
                statistics.invalid_commands += 1
                feedback = {
                    'frame': frame,
                    'requested_view': {
                        'resolution_level': requested.resolution_level,
                        'center_x': requested.center_x,
                        'center_y': requested.center_y,
                    },
                    'reason': str(exc),
                }
        else:
            feedback = None

    coco_map_50, ap_by_class = score(scene, predictions)
    print("AP@0.50 by class")
    for name, value in sorted(ap_by_class.items(), key=lambda item: -item[1]):
        print(f"  {name:16s} {value:.3f}")
    print()
    print(f"COCO mAP@0.50: {coco_map_50:.6f}")
    return coco_map_50

if __name__ == "__main__":
    run_direct()
