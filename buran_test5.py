import rospy
from clover import srv
from std_srvs.srv import Trigger
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
from sensor_msgs.msg import Range
import numpy as np
from aruco_pose.msg import MarkerArray

rospy.init_node('buran')

get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
land = rospy.ServiceProxy('land', Trigger)

debug = rospy.Publisher("/main_camera/debug", Image, queue_size=1)

# constants
OBJ_AREA = 80
IS_GAZEBO = False

# dictionaries
if IS_GAZEBO:
    ranges = {
        'red': (np.array([169, 100, 155]), np.array([179, 255, 255]), np.array([0, 100, 155]), np.array([15, 255, 255])),
        'yellow': (np.array([20, 80, 120]), np.array([40, 255, 255])),
        'green': (np.array([50, 117, 120]), np.array([70, 255, 255])),
        'blue': (np.array([110, 70, 115]), np.array([130, 255, 255]))
    }
else:
    ranges = {
        'red': (np.array([170, 30, 120]), np.array([179, 255, 255]), np.array([0, 30, 120]), np.array([10, 255, 255])),
        'yellow': (np.array([23, 20, 170]), np.array([35, 255, 255])),
        'green': (np.array([60, 50, 80]), np.array([80, 255, 255])),
        'blue': (np.array([85, 110, 100]), np.array([110, 255, 255]))
    }

cargos_colors_quantity = {
    'red' : 0,
    'yellow': 0,
    'green': 0,
    'blue': 0
}

cargos_type_num = {
    'red' : 3,
    'yellow': 0,
    'green': 1,
    'blue': 2
}

cargos_type = {
    'red' : 'correspondence',
    'yellow': 'products',
    'green': 'clothes',
    'blue': 'fragile packaging'
}

def navigate_wait(x=0, y=0, z=0, yaw=math.radians(90), speed=0.5, frame_id='aruco_map', auto_arm=False, tolerance=0.15):
    navigate(x=x, y=y, z=z, yaw=yaw, speed=speed, frame_id=frame_id, auto_arm=auto_arm)

    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
            break
        rospy.sleep(0.2)

def detectCargos(data, side):
    frame = CvBridge().imgmsg_to_cv2(data, 'bgr8') #[90:150,120:200]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out = frame.copy()
    auxFrame = frame.copy()
    out_mask = cv2.cvtColor(np.zeros(out.shape, np.uint8), cv2.COLOR_BGR2GRAY)


    linePoints = getDevideLine()

    kernal = np.ones((5,5),np.uint8)

    for key in ranges:  # for each masks
        if len(ranges[key]) == 4:
            mask = cv2.morphologyEx(cv2.inRange(hsv, ranges[key][0], ranges[key][1]) + cv2.inRange(hsv, ranges[key][2], ranges[key][3]), cv2.MORPH_OPEN, kernal)
        else:
            mask = cv2.morphologyEx(cv2.inRange(hsv, ranges[key][0], ranges[key][1]), cv2.MORPH_OPEN, kernal)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        contours = [i for i in contours if cv2.contourArea(i) > OBJ_AREA]
        
        #print(str(len(contours)) + key)
        curContours = []
        for cnt in contours:
            [x, y, w, h] = cv2.boundingRect(cnt)
            if side == 'left':
                if x + w//2 < linePoints[0][0]:
                    curContours.append(cnt)
            else:
                if x + w//2 > linePoints[0][0]:
                   curContours.append(cnt)
        contours = curContours
        #print(str(len(contours)) + key)
        #     
        cargos_colors_quantity[key] += len(contours)


        for cnt in contours: #draw contours
            cv2.drawContours(out, [cnt], -1, (255, 255, 255), 2)
            [x, y, w, h] = cv2.boundingRect(cnt)
            cv2.putText(out, key, (x - 15, y + h + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,255,255), 1)
            cv2.putText(out_mask, key, (x - 15, y + h + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255,255,255), 1)
        
        out_mask = cv2.bitwise_or(out_mask, mask)

    auxFrame = cv2.line(auxFrame, linePoints[0],linePoints[1], (0,0,0), 2)
    if side == 'left': dx = -20
    else: dx = 20
    for k in range(1, 3):
        auxFrame = cv2.arrowedLine(auxFrame , (linePoints[0][0], 80*k), (linePoints[0][0] + dx, 80*k), (0,0,0), 2)
    
    out = cv2.bitwise_and(out, out, mask=out_mask)
    vis = np.concatenate((out, auxFrame), axis=1)
    debug.publish(CvBridge().cv2_to_imgmsg(vis, 'bgr8'))

def getDevideLine():
    markers = rospy.wait_for_message('aruco_detect/markers', MarkerArray).markers
    for marker in markers:
        if marker.id == 22 or marker.id == 27 or marker.id == 32:
            x = int(marker.c1.x)
            return (x, 0), (x, 240)

#flight
navigate(z=1.5, frame_id='body', auto_arm=True)
rospy.sleep(3)
navigate_wait(z=1.8)

navigate_wait(x=0.9, y=4.5, z=1.8)
navigate_wait(x=0.8, y=4.5, z=1.4)
rospy.sleep(3)

detectCargos(rospy.wait_for_message('main_camera/image_raw_throttled', Image), 'left')

navigate_wait(x=2.8, y=4.5, z=1.4)
rospy.sleep(3)

detectCargos(rospy.wait_for_message('main_camera/image_raw_throttled', Image), 'right')

navigate_wait(x=2.8, y=4.5, z=1.8)

#report 1
balance = 0
for col in cargos_colors_quantity:
    balance += cargos_colors_quantity[col]
print('Balance %i cargo' % balance)
for i in range(len(cargos_type_num)):
    for t in cargos_type_num:
        if cargos_type_num[t] == i:
            print('Type %i: %i cargo' % (i, cargos_colors_quantity[t]))
print('')


navigate_wait(z=1.8)
land()