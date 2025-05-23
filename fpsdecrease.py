import cv2

input_video_path = './data/yunus.mp4'
output_video_path = './clean_data/yunus_clean.mp4'

cap = cv2.VideoCapture(input_video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_60 = cap.get(cv2.CAP_PROP_FPS)

new_fps = 3

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, new_fps, (frame_width, frame_height))

frame_counter = 0
while True:
    ret, frame = cap.read()
    print(frame_counter)
    if not ret:
        break

    if frame_counter % int(fps_60 / new_fps) == 0:
        out.write(frame)
    
    frame_counter += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("Video converting successfuly.")
