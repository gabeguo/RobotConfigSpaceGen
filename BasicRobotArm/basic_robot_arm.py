import pybullet as p
import time
import pybullet_data
import math

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

planeID = p.loadURDF('plane.urdf')

boxID = p.loadURDF('box.urdf')

basic_arm = p.loadURDF('basic_arm.urdf', [0, 0, 0], useFixedBase = 1)

print('planeID', planeID)
print('boxID', boxID)
print('basic arm', basic_arm)

n_joints = p.getNumJoints(basic_arm)
print('n joints', n_joints)

'''
position, orientation = p.getBasePositionAndOrientation(basic_arm)
print('position:', position)
print('orientation:', orientation)

joint_info_first = p.getJointInfo(basic_arm, 0)
print('joint info', joint_info_first)
joint_info_second = p.getJointInfo(basic_arm, 1)
print('joint info', joint_info_second)

joint_positions = [j[0] for j in p.getJointStates(basic_arm, range(n_joints))]
print('joint positions', joint_positions)
'''

p.setGravity(0, 0, -10)

#time.sleep(3)

p.setRealTimeSimulation(0)

p.setJointMotorControlArray(basic_arm, range(n_joints), p.POSITION_CONTROL, targetPositions=[math.pi, math.pi / 2, math.pi / 2])
for _ in range(100):
    p.stepSimulation()
    if len(p.getContactPoints(bodyA=basic_arm, bodyB=boxID)) > 0:
        print('collision')
        #time.sleep(1)
    time.sleep(1./100.)

p.setJointMotorControlArray(basic_arm, range(n_joints), p.POSITION_CONTROL, targetPositions=[math.pi * 2, math.pi / 4, math.pi / 4])
for _ in range(500):
    p.stepSimulation()
    if len(p.getContactPoints(bodyA=basic_arm, bodyB=boxID)) > 0:
        print('collision')
        #time.sleep(1)
    time.sleep(1./100.)

#time.sleep(3)
