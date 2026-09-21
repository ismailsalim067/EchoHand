import pybullet as p
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
plane_id = p.loadURDF("plane.urdf")

while True:
    p.stepSimulation()