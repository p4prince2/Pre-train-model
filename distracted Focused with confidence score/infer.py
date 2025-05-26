# import cv2
# import torch
# import numpy as np
# from model import FocusLSTM
# from feature_extractor import extract_features_from_frame
#
# def main():
#     # Load trained model
#     model = FocusLSTM()
#     model.load_state_dict(torch.load("model.pth"))  # Make sure the name matches saved file
#     model.eval()
#
#     cap = cv2.VideoCapture(0)
#     sequence = []
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame.")
#             break
#
#         # Extract features from frame (e.g., head pose, eye position, etc.)
#         feat = extract_features_from_frame(frame)
#
#         if feat is not None:
#             sequence.append(feat)
#
#             # Maintain last 30 frames
#             if len(sequence) > 30:
#                 sequence.pop(0)
#
#             # Predict only when 30-frame sequence is available
#             if len(sequence) == 30:
#                 input_seq = torch.tensor([sequence], dtype=torch.float32)
#
#                 with torch.no_grad():
#                     out = model(input_seq)
#                     pred = torch.argmax(out).item()
#                     confidence = torch.softmax(out, dim=1)[0][pred].item()
#
#                     label = "Focused" if pred == 1 else "Distracted"
#                     text = f"{label} ({confidence:.2f})"
#
#                     # Display label on frame
#                     cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                                 1.5, (0, 255, 0) if pred == 1 else (0, 0, 255), 3)
#         else:
#             print("Could not extract features from this frame.")
#
#         # Show the frame
#         cv2.imshow("Driver Monitoring", frame)
#
#         # Exit if ESC key is pressed
#         if cv2.waitKey(1) & 0xFF == 27:
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     main()




import cv2
import torch
import numpy as np
from model import FocusLSTM
from feature_extractor import extract_features_from_frame

def main():
    # Load trained model
    model = FocusLSTM()
    model.load_state_dict(torch.load("model.pth"))  # Ensure this matches your saved model name
    model.eval()

    # Load video file instead of webcam
    video_path = "sample_video.mp4"  # Update with your actual video file path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to grab frame.")
            break

        # Extract features from current frame
        feat = extract_features_from_frame(frame)

        if feat is not None:
            sequence.append(feat)

            # Keep only the last 30 frames
            if len(sequence) > 30:
                sequence.pop(0)

            # Make prediction when sequence is ready
            if len(sequence) == 30:
                input_seq = torch.tensor([sequence], dtype=torch.float32)

                with torch.no_grad():
                    out = model(input_seq)
                    pred = torch.argmax(out).item()
                    confidence = torch.softmax(out, dim=1)[0][pred].item()

                    label = "Focused" if pred == 1 else "Distracted"
                    text = f"{label} ({confidence:.2f})"

                    # Draw label on frame
                    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 255, 0) if pred == 1 else (0, 0, 255), 3)
        else:
            print("Warning: Could not extract features from frame.")

        # Show the annotated video
        cv2.imshow("Driver Monitoring - Video", frame)

        # Press ESC to quit
        if cv2.waitKey(25) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
