# FlowSense

**AI-Driven Adaptive Traffic Signal System with Emergency Green Corridors**

FlowSense is an intelligent traffic management system that uses computer vision and adaptive signal control to dynamically optimize traffic signals based on real-time traffic conditions. The system also supports priority green corridors for emergency vehicles such as ambulances and fire services.

The project demonstrates how **edge AI + decentralized traffic control + predictive corridor scheduling** can reduce congestion and enable faster emergency response in dense urban environments like Delhi.

---

# Problem

Urban traffic systems rely largely on **static or pre-timed signal schedules** that do not adapt to real-time traffic conditions.

This leads to:

- unnecessary waiting at empty intersections  
- congestion buildup and spillback  
- inefficient vehicle throughput  
- delayed emergency vehicle movement  

Emergency corridors today often require **manual traffic police intervention**, which is difficult to scale.

A scalable solution requires **real-time traffic perception and adaptive decision making at intersections.**

---

# Solution

FlowSense introduces a **three-layer intelligent traffic control architecture**.

## 1. Local Intersection Intelligence

Each intersection uses camera feeds and computer vision to generate a **real-time traffic state vector**.

The system:

- detects vehicles  
- estimates queue lengths  
- dynamically adjusts signal phases using **pressure-based adaptive control**

---

## 2. Regional Coordination

Neighboring intersections share traffic flow information to:

- align green phases
- maintain platoon movement across corridors
- reduce stop-and-go traffic

---

## 3. Emergency Corridor Scheduler

When an **ambulance is detected**, the system:

1. Predicts the vehicle route  
2. Creates temporary green windows across intersections  
3. Forms a **moving green corridor**

This allows emergency vehicles to pass through intersections with **minimal delay** while maintaining safe signal transitions.

---

# System Architecture

```
Traffic Cameras
        │
        ▼
Computer Vision Detection (YOLOv8)
        │
        ▼
Vehicle Tracking & Zone Analysis
        │
        ▼
Intersection State Vector
(queue length, arrivals, departures)
        │
        ▼
Local Adaptive Signal Controller
(pressure-based phase selection)
        │
        ├──────────► Regional Coordination Layer
        │              (platoon alignment)
        │
        ▼
Emergency Corridor Scheduler
(ambulance priority routing)
        │
        ▼
Traffic Signal Execution
```

---

# Setup

## Install Dependencies

```bash
pip install ultralytics opencv-python numpy
```

---

# Run the Detection Pipeline

```bash
python src/detect_track.py
```

This runs **vehicle detection and tracking** on the traffic video.

---

# Generate Traffic State

```bash
python src/state_estimator.py
```

Outputs **per-second traffic state logs**.

---

# Run Adaptive Controller Simulation

```bash
python src/controller.py
```

This simulates **traffic signal decision making** based on the detected traffic state.

---
