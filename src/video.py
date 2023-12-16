import cv2

def calculate_velocity(video_path):
    # Step 2: Read the video
    cap = cv2.VideoCapture(video_path)

    # Step 3: Object Tracking
    tracker = cv2.Tracker()
    success, frame = cap.read()
    bbox = cv2.selectROI("Tracking", frame, False)
    tracker.init(frame, bbox)

    velocities = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(i) for i in bbox]
            # Calculate velocity based on the change in position (e.g., x, y) over time
            velocity = calculate_object_velocity(x, y)
            velocities.append(velocity)

        # Display the frame with tracking information
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

    return velocities

def calculate_object_velocity(x, y):
    # You can implement your velocity calculation logic here based on the change in position over time.
    # This is just a placeholder; you may need to customize it based on your specific use case.
    velocity = 0
    return velocity


# Example usage:
video_path = 'recordec_video.mp4'
velocities = calculate_velocity(video_path)
print("Velocities:", velocities)
