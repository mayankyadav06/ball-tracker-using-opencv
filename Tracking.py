# Import necessary libraries
import cv2
import numpy as np
import time
import gdown

# Download the video file from Google Drive
url = 'https://drive.google.com/uc?id=1goI3aHVE29Gko9lpTzgi_g3CZZPjJq8w'
output = 'AI_Assignment_video.mp4'
gdown.download(url, output, quiet=False)

# Open the video file
cap = cv2.VideoCapture(output)

# Get the video's width, height, and fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter('Tracking.mp4', fourcc, fps, (width, height))


try:
    # Record the start time
    start_time = time.time()

    # Define the color ranges for different balls
    color_ranges = {
        "Orange": ([5, 80, 180], [10, 255, 255]),
        # "White": ([0, 20, 140], [30, 50, 220]),
        "Yellow": ([20, 100, 100], [30, 255, 255]),
        "Green": ([50, 40, 50], [90, 255, 150])
    }


    # Define the quadrants
    quadrants = [
        ((550, 275), (760, 575)),  # Q1
        ((350, 275), (550, 575)),  # Q2
        ((350, 0), (550, 275)),    # Q3
        ((550, 0), (760, 275))     # Q4
    ]

    # Function to get the quadrant based on x, y coordinates
    def get_quadrant(x, y):
        for i, ((x1, y1), (x2, y2)) in enumerate(quadrants):
            if x1 <= x < x2 and y1 <= y < y2:
                return i + 1
        return None

    # Dictionaries to store ball information
    ball_quadrants = {}
    ball_entry_time = {}
    ball_exit_time = {}

    # Function to update ball information
    def update_ball_info(color, quadrant, status):
        # Calculate the elapsed time since the start of the video
        timestamp = time.time() - start_time
        # Convert the timestamp to minutes and seconds
        minutes, seconds = divmod(int(timestamp), 60)
        if seconds < 3:
            minutes -= 1
            seconds += 60
        
        # Check if the color is not in ball_quadrants or if the ball has moved to a new quadrant
        if color not in ball_quadrants or ball_quadrants[color][0] != quadrant:
            # If the color is not in ball_entry_time, record the current time as the entry time
            if color not in ball_entry_time:
                ball_entry_time[color] = time.time()
            
            # If the ball has been in the quadrant for more than 3 seconds, update ball_quadrants and write the entry event to the file
            elif time.time() - ball_entry_time[color] >= 3:
                ball_quadrants[color] = (quadrant, minutes, seconds - 3, status)
                
                with open('Tracking.txt', 'a') as f:
                    f.write(f"{color}: {status}, quadrant {quadrant}, {minutes}:{seconds - 2}\n")
        
        # If the ball is in the same quadrant but its status has changed
        elif color in ball_quadrants and ball_quadrants[color][0] == quadrant and ball_quadrants[color][3] != status:
            
            # If the color is not in ball_exit_time, record the current time as the exit time
            if color not in ball_exit_time:
                ball_exit_time[color] = time.time()
        
            # If the ball has been exiting for more than 3 seconds, update ball_quadrants and write the exit event to the file
            elif time.time() - ball_exit_time[color] >= 2:
                ball_quadrants[color] = (ball_quadrants[color][0], minutes, seconds - 3, status)
                
                with open('Tracking.txt', 'a') as f:
                    f.write(f"{color}: {status}, quadrant {ball_quadrants[color][0]}, {minutes}:{seconds - 2}\n")


    # Main loop to process each frame of the video
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True and frame is not None:
            frame = cv2.resize(frame, (800, 600))
            hsv =  cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            
            current_colors = []
            # Mask to identify and capture the balls
            for color, (lower, upper) in color_ranges.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                mask = cv2.erode(mask, None, iterations=1)
                mask = cv2.dilate(mask, None, iterations=2)
                contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                
                # Detecting the balls and creating chain with pixels which have the same colour
                for contour in contours:
                    if 1000 < cv2.contourArea(contour) <= 12000:
                        ((x, y), radius) = cv2.minEnclosingCircle(contour)
                        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                        cv2.putText(frame, color, (int(x) - 20, int(y) - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,0), 2)
                        
                        quadrant = get_quadrant(x, y)
                        update_ball_info(color, quadrant, 'Entry')
                        current_colors.append(color)
                        
            for color in set(ball_quadrants.keys()) - set(current_colors):
                _, minutes, seconds, status = ball_quadrants[color]
                if status != 'Exit':
                    update_ball_info(color, ball_quadrants[color][0], 'Exit')
            
            for i, (color, (quadrant, minutes, seconds, status)) in enumerate(ball_quadrants.items()):
                cv2.putText(frame, f"{color}: {status}, quadrant {quadrant}, {minutes}:{seconds}", (10, 30 + i*30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0,0,0), 2)
            
            if frame is not None:
                out.write(frame)
            cv2.imshow("",frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
finally:         
    cap.release()
    out.release()
    cv2.destroyAllWindows()
