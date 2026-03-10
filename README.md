# FlowSense
### AI-Driven Adaptive Traffic Signal System with Emergency Green Corridors

FlowSense is an intelligent traffic management system that uses **computer vision and adaptive signal control** to dynamically optimize traffic signals based on real-time traffic conditions.

The system also supports **priority green corridors for emergency vehicles** such as ambulances and fire services.

The project demonstrates how **edge AI + decentralized traffic control + predictive corridor scheduling** can reduce congestion and enable faster emergency response in dense urban environments like **Delhi**.

---

## Problem

Urban traffic systems rely largely on **static or pre-timed signal schedules** that do not adapt to real-time traffic conditions.

This leads to:

- unnecessary waiting at empty intersections  
- congestion buildup and spillback  
- inefficient vehicle throughput  
- delayed emergency vehicle movement  

Emergency corridors today often require **manual traffic police intervention**, which is difficult to scale.

A scalable solution requires **real-time traffic perception and adaptive decision making at intersections**.

---

## Solution

FlowSense introduces a **three-layer intelligent traffic control architecture**.

### 1. Local Intersection Intelligence

Each intersection uses camera feeds and computer vision to generate a **real-time traffic state vector**.

The system:

- detects vehicles  
- estimates queue lengths  
- dynamically adjusts signal phases using **pressure-based adaptive control**

---

### 2. Regional Coordination

Neighboring intersections share traffic flow information to:

- align green phases  
- maintain **platoon movement across corridors**  
- reduce stop-and-go traffic

---

### 3. Emergency Corridor Scheduler

When an ambulance is detected, the system:

1. predicts its route through the road network  
2. estimates arrival times at upcoming intersections  
3. creates temporary **green windows across intersections**

This forms a **moving green corridor** that allows emergency vehicles to pass with minimal delay while maintaining safe signal transitions.

---

## System Architecture
