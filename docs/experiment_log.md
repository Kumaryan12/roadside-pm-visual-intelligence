# Experiment Log

Use this file to record each experiment.

## Format

### Experiment Name
Date:
Dataset:
Features:
Model:
Validation:
Metrics:
Result:
Conclusion:

---

## Experiment: YOLO11m IDD15 Indian Vehicle Detector v1

### Date
2026-05-28

### Objective
Fine-tune a YOLO11m object detector on an Indian road-scene vehicle dataset to address limitations of COCO-pretrained YOLO models, especially the absence of an explicit auto-rickshaw class.

### Motivation
The default YOLO detector does not reliably detect Indian auto-rickshaws because auto-rickshaw is not a standard COCO class. Since auto-rickshaws are common in Indian traffic and relevant for emission-aware PM estimation, an Indian-road-specific detector is required.

### Dataset
Indian Driving Dataset - Detections YOLOv11 version.

Dataset classes: 15

Classes:
1. animal
2. autorickshaw
3. bicycle
4. bus
5. car
6. caravan
7. motorcycle
8. person
9. rider
10. traffic light
11. traffic sign
12. trailer
13. train
14. truck
15. vehicle fallback

Dataset split:
- Train images: 33,569
- Validation images: 4,196
- Test images: 4,197

### Model
Base model: YOLO11m pretrained weights

Training type: Fine-tuning

Initial pretrained model:
- YOLO11m pretrained detector

Final model:
- yolo11m_idd15_v1_best.pt

### Training Environment
Platform: Kaggle Notebook

GPU:
- Tesla T4 GPU 0
- Tesla T4 GPU 1

Training used both GPUs through Ultralytics DDP.

Python version:
- Python 3.12.13

Ultralytics version:
- 8.4.56

Torch version:
- torch 2.10.0 + cu128

### Training Configuration
Image size: 640

Batch size: 16

Device:
- device=[0, 1]

Workers: 4

AMP: Enabled

Save period: 5 epochs

Patience: 10

Training run name:
- yolo11m_idd15_v1_continue_from_epoch15

Training method:
- Continued fine-tuning from previously saved epoch-15 checkpoint weights.
- True optimizer-state resume failed due to optimizer parameter-group mismatch.
- Therefore, last_new.pt was used as pretrained weights with resume=False and correct IDD data YAML.

### Important Note on Resume
A true resume attempt failed because Ultralytics tried to load incompatible optimizer state and also incorrectly fell back to coco8.yaml / 80-class configuration. To avoid corrupting the experiment, the checkpoint was used only as pretrained weights and the model was trained again with the correct 15-class IDD data.yaml.

### Dataset Issues Observed
YOLO reported:
- 1 corrupt training image/label ignored due to non-normalized or out-of-bounds coordinates.
- A few duplicate labels were automatically removed.
- Kaggle input directory was read-only, so label cache was not saved.

These issues were not fatal and training completed successfully.

### Training Outcome
Training stopped early due to early stopping.

Reason:
- No validation improvement for 10 epochs.
- Best result observed at epoch 1 of the continued run.

Total continued epochs completed:
- 11 epochs

### Overall Validation Metrics
Overall Precision: 0.677

Overall Recall: 0.488

Overall mAP50: 0.497

Overall mAP50-95: 0.330

### Class-wise Validation Metrics

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| animal | 0.533 | 0.355 | 0.360 | 0.182 |
| autorickshaw | 0.779 | 0.702 | 0.743 | 0.542 |
| bicycle | 0.610 | 0.478 | 0.498 | 0.310 |
| bus | 0.791 | 0.671 | 0.722 | 0.559 |
| car | 0.760 | 0.649 | 0.693 | 0.494 |
| caravan | 0.373 | 0.611 | 0.372 | 0.349 |
| motorcycle | 0.746 | 0.672 | 0.698 | 0.426 |
| person | 0.679 | 0.500 | 0.534 | 0.303 |
| rider | 0.744 | 0.550 | 0.597 | 0.353 |
| traffic light | 0.552 | 0.422 | 0.428 | 0.231 |
| traffic sign | 0.582 | 0.429 | 0.421 | 0.237 |
| train | 1.000 | 0.000 | 0.020 | 0.006 |
| truck | 0.715 | 0.649 | 0.694 | 0.512 |
| vehicle fallback | 0.620 | 0.147 | 0.182 | 0.109 |

### Key PM-Relevant Results
Auto-rickshaw performance:
- Precision: 0.779
- Recall: 0.702
- mAP50: 0.743
- mAP50-95: 0.542

Motorcycle performance:
- Precision: 0.746
- Recall: 0.672
- mAP50: 0.698

Bus performance:
- Precision: 0.791
- Recall: 0.671
- mAP50: 0.722

Truck performance:
- Precision: 0.715
- Recall: 0.649
- mAP50: 0.694

Car performance:
- Precision: 0.760
- Recall: 0.649
- mAP50: 0.693

### Interpretation
The fine-tuned YOLO11m model successfully learns Indian traffic categories, especially auto-rickshaw, which is absent from default COCO-trained YOLO models.

This model is expected to improve PM-relevant traffic feature extraction because it can explicitly detect:
- auto-rickshaws
- motorcycles
- buses
- trucks
- cars
- riders/persons

The most important improvement is the explicit auto-rickshaw class, with mAP50 = 0.743 and recall = 0.702.

### Limitations
- Validation is on IDD validation data, not yet on our own roadside camera frames.
- Domain shift may exist between IDD images and our PM-density project frames.
- Vehicle fallback class has weak recall and should be treated cautiously.
- Train class is irrelevant for our PM task and has very few samples.
- The detector must still be evaluated on our actual video frames before being trusted for PM modeling.

### Saved Outputs
Kaggle output folder:
- /kaggle/working/detector_training/yolo11m_idd15_v1_continue_from_epoch15

Important files:
- weights/best.pt
- weights/last.pt
- results.csv
- results.png
- confusion_matrix.png
- confusion_matrix_normalized.png
- BoxPR_curve.png
- BoxF1_curve.png
- BoxP_curve.png
- BoxR_curve.png

Local model filename:
- models/detectors/yolo11m_idd15_v1_best.pt

Note:
The .pt model file is not committed to GitHub because model weights are ignored by .gitignore.

### Next Step
Run this fine-tuned detector on the PM-density project’s processed frames and compare it against the old COCO YOLO detector.

Comparison metrics:
- total vehicle count difference
- auto-rickshaw count
- motorcycle count
- bus count
- truck count
- traffic-load score difference
- emission-proxy difference
- downstream PM/effective-density model performance


---

## Experiment Update: Vehicle Detection v1 Local Frame Verification

### Date
2026-05-28

### Objective
Verify whether the fine-tuned IDD YOLO11m detector works on the PM-density project's own processed traffic frames.

### Result
The detector successfully detected PM-relevant vehicle classes on local processed frames, including explicit auto-rickshaw detections. Vehicle-only filtering was applied so non-vehicle classes such as person, rider, traffic light, traffic sign, animal, and train were excluded from inference outputs and annotated images.

### Current Vehicle Classes Used
- autorickshaw
- bicycle
- bus
- car
- caravan
- motorcycle
- trailer
- truck
- vehicle fallback

### Ignored Classes
- person
- rider
- animal
- traffic light
- traffic sign
- train

### Status
Vehicle Detection v1 is working and ready for full-frame feature extraction.

### Next Step
Run full inference on all processed frames and save:
outputs/features/idd_vehicle_detections_processed_frames.csv

After this, begin road-condition and road-dust feature extraction as a separate module.
