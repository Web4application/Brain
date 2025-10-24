git clone https://github.com/Web4application/Brain.git
cd Brain

git clone https://github.com/Web4application/EDQ-AI.git
cd EDQ-AI

git clone https://github.com/Web4application/SERAI.git
cd SERAI


+-------------------+
|   Arduino Board   |
|-------------------|
| Sensors: Temp,   |
| Motion, Light,   |
| Distance, etc.   |
|                   |
| Actuators: Motors, LEDs, Relays |
+-------------------+
          │
          │  Sensor Data / Control Signals
          ▼
+-------------------+
|     EDQ AI        |
|-------------------|
| Data Processing   |
| Filtering /       |
| Aggregation       |
+-------------------+
          │
          │  Structured & Clean Data
          ▼
+-------------------+
|      SERAI AI     |
|-------------------|
| Reasoning Engine  |
| Simulation        |
| Predictive Models |
| Decision Making   |
+-------------------+
          │
          │  Commands / Actions
          ▼
+-------------------+
|   Arduino Board   |
| (Execution Layer) |
| Motors, LEDs,     |
| Relays, etc.      |
+-------------------+
          │
          ▼
      Real World
