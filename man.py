brain = NeurobotBrain()
optimizer = optim.Adam(brain.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Camera setup
cap = cv2.VideoCapture(0)

try:
    while True:
        # ----- Read Sensors -----
        if ser.in_waiting:
            data = ser.readline().decode().strip()
            if data:
                distance, imu = map(float, data.split(','))  # Arduino sends "distance,imu_angle"
                sensor_input = torch.tensor([[distance, imu]], dtype=torch.float32)
        
        # ----- Read Camera -----
        ret, frame = cap.read()
        if not ret:
            continue
        image_input = preprocess_camera(frame)
        
        # ----- ANN Decision -----
        action = choose_action(brain, image_input, sensor_input)
        ser.write((action + "\n").encode())
        
        # ----- Reward & Training -----
        reward = compute_reward(distance)
        target = torch.zeros(1, 3)
        target[0, actions.index(action)] = reward
        
        optimizer.zero_grad()
        output = brain(image_input, sensor_input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        print(f"Distance: {distance:.1f} | IMU: {imu:.1f} | Action: {action} | Reward: {reward}")

except KeyboardInterrupt:
    print("Stopping Neurobot brain...")
    cap.release()
    ser.close()
