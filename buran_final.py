# import libraries
import rospy
from clover import srv
from std_srvs.srv import Trigger
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np
from aruco_pose.msg import MarkerArray
from mavros_msgs.srv import CommandBool
from clover.srv import SetLEDEffect

rospy.init_node('buran')

# services
get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
land = rospy.ServiceProxy('land', Trigger)
arming = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
set_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect)

debug = rospy.Publisher("/main_camera/debug", Image, queue_size=1)

# constants
IS_GAZEBO = False

# dictionaries
if IS_GAZEBO:
    ranges = {
        'red': (
            np.array([169, 100, 155]), np.array([179, 255, 255]), np.array([0, 100, 155]), np.array([15, 255, 255])),
        'yellow': (np.array([20, 80, 120]), np.array([40, 255, 255])),
        'green': (np.array([50, 117, 120]), np.array([70, 255, 255])),
        'blue': (np.array([110, 70, 115]), np.array([130, 255, 255]))
    }
else:
    ranges = {
        'red': (np.array([170, 30, 120]), np.array([179, 255, 255]), np.array([0, 30, 120]), np.array([10, 255, 255])),
        # 'yellow': (np.array([23, 20, 170]), np.array([35, 255, 255])),
        'yellow': (np.array([25, 30, 170]), np.array([50, 255, 255])),
        'green': (np.array([60, 50, 80]), np.array([80, 255, 255]), np.array([50, 10, 90]), np.array([65, 30, 255])),
        # 'blue': (np.array([85, 110, 100]), np.array([110, 255, 255]))
        'blue': (np.array([90, 90, 90]), np.array([110, 255, 255]))
    }

cargos_colors_quantity = {  # colors and their quantity
    'red': 0,
    'yellow': 0,
    'green': 0,
    'blue': 0
}

cargos_type_num = {  # colors and their type number
    'red': 3,
    'yellow': 0,
    'green': 1,
    'blue': 2
}

cargos_type = {  # colors and their type
    'red': 'correspondence',
    'yellow': 'products',
    'green': 'clothes',
    'blue': 'fragile packaging'
}

cargos_led_colors = {  # colors and their colors for led
    'red': (255, 0, 0),
    'yellow': (255, 255, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255)
}

digit_rec = {    # dict for digit recognition
    0: (0, 6),
    1: (1, 4),
    2: (2, 7, 5),
    3: (3, 8, 9)
}


def navigate_wait(x=0, y=0, z=0, yaw=math.radians(90), speed=0.5, frame_id='aruco_map', auto_arm=False, tolerance=0.15):
    navigate(x=x, y=y, z=z, yaw=yaw, speed=speed,
             frame_id=frame_id, auto_arm=auto_arm)

    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
            break
        rospy.sleep(0.2)

# function for detect cargos
def detectCargos(data, side):
    frame = CvBridge().imgmsg_to_cv2(data, 'bgr8')  # [90:150,120:200]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out = frame.copy()
    auxFrame = frame.copy()
    out_mask = cv2.cvtColor(np.zeros(out.shape, np.uint8), cv2.COLOR_BGR2GRAY)

    linePoints = getDevideLine()

    kernal = np.ones((5, 5), np.uint8)

    for key in ranges:  # for each color get mask
        if len(ranges[key]) == 4:
            mask = cv2.morphologyEx(cv2.inRange(hsv, ranges[key][0], ranges[key][1]) + cv2.inRange(
                hsv, ranges[key][2], ranges[key][3]), cv2.MORPH_OPEN, kernal)
        else:
            mask = cv2.morphologyEx(cv2.inRange(
                hsv, ranges[key][0], ranges[key][1]), cv2.MORPH_OPEN, kernal)

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        contours = [i for i in contours if cv2.contourArea(i) > 80]  # OBJ_AREA

        #print(str(len(contours)) + key)
        curContours = []
        for cnt in contours:  # impose conditions on the contours
            [x, y, w, h] = cv2.boundingRect(cnt)
            if side == 'left':
                if x + w // 2 < linePoints[0][0]:
                    curContours.append(cnt)
            else:
                if x + w // 2 > linePoints[0][0]:
                    curContours.append(cnt)
        contours = curContours
        # print(str(len(contours)) + key)
        cargos_colors_quantity[key] += len(contours)  # recording the recognition results

        for cnt in contours:  # draw contours
            cv2.drawContours(out, [cnt], -1, (255, 255, 255), 2)
            [x, y, w, h] = cv2.boundingRect(cnt)
            cv2.putText(out, key, (x - 15, y + h + 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 1)
            cv2.putText(out_mask, key, (x - 15, y + h + 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (255, 255, 255), 1)

        out_mask = cv2.bitwise_or(out_mask, mask)

    auxFrame = cv2.line(auxFrame, linePoints[0], linePoints[1], (0, 0, 0), 2)
    if side == 'left':
        dx = -20
    else:
        dx = 20
    for k in range(1, 3):
        auxFrame = cv2.arrowedLine(
            auxFrame, (linePoints[0][0], 80 * k), (linePoints[0][0] + dx, 80 * k), (0, 0, 0), 2)

    out = cv2.bitwise_and(out, out, mask=out_mask)
    vis = np.concatenate((out, auxFrame), axis=1)
    debug.publish(CvBridge().cv2_to_imgmsg(vis, 'bgr8'))

# function for get coordinates for devide line
def getDevideLine():
    markers = rospy.wait_for_message(
        'aruco_detect/markers', MarkerArray).markers
    for marker in markers:
        if marker.id == 27:
            x = int(marker.c1.x)
            return (x, 0), (x, 240)

# function for digits recognition
def detectDigits():
    digits = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    samples = np.loadtxt('evgsamples.data', np.float32)
    responses = np.loadtxt('evgresponses.data', np.float32)
    responses = responses.reshape((responses.size, 1))
    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses) # load model and samples

    for _ in range(10):  # do recognition 10 times for more accuracy
        frame = rospy.wait_for_message('/main_camera/image_raw', Image)
        im = CvBridge().imgmsg_to_cv2(frame, 'bgr8')[40:200, 80:240]

        out = np.zeros(im.shape, np.uint8)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        kernal = np.ones((3, 3), np.uint8)
        if IS_GAZEBO:
            mask = cv2.morphologyEx(cv2.inRange(hsv, np.array(
                [55, 110, 110]), np.array([65, 255, 255])), cv2.MORPH_OPEN, kernal)
        else:
            mask = cv2.morphologyEx(cv2.inRange(hsv, np.array(
                [60, 50, 70]), np.array([85, 255, 255])), cv2.MORPH_OPEN, kernal)

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        if len(contours) != 0:
            cnt = max(contours, key=cv2.contourArea)  # get max contour on frame
            [x, y, w, h] = cv2.boundingRect(cnt)
            if h > 25:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # roi = mask[y:y + h, x:x + w]
                try:  # conversions to the template form
                    roi = mask[y:y + h, x + w // 2 -
                               h // 2:x + w // 2 + h // 2]
                    roismall = cv2.resize(roi, (20, 20))
                    roismall = roismall.reshape((1, 400))
                    roismall = np.float32(roismall)
                    retval, results, neigh_resp, dists = model.findNearest(  #the main function of the knn algorithm
                        roismall, k=1)
                    string = str(int((results[0][0])))
                    # print(string)
                    digits[int(string)] += 1
                    cv2.putText(out, string, (x, y + h), 0, 1, (0, 255, 0))
                except cv2.error as e:
                    print('Invalid frame!')

        debug.publish(CvBridge().cv2_to_imgmsg(mask, 'mono8'))
        # debug.publish(CvBridge().cv2_to_imgmsg(out, 'bgr8'))
        # rospy.sleep(0.5)

    print(digits)
    return digits.index(max(digits))

# this function return true if drone can see the contour, else - false
def isDronePointHere():
    frame = rospy.wait_for_message('/main_camera/image_raw', Image)
    im = CvBridge().imgmsg_to_cv2(frame, 'bgr8')[0:240, 60:260]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    kernal = np.ones((3, 3), np.uint8)
    if IS_GAZEBO:
        mask = cv2.morphologyEx(cv2.inRange(hsv, np.array(
            [55, 110, 110]), np.array([65, 255, 255])), cv2.MORPH_OPEN, kernal)
    else:
        mask = cv2.morphologyEx(cv2.inRange(hsv, np.array(
            [60, 50, 70]), np.array([85, 255, 255])), cv2.MORPH_OPEN, kernal)

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    debug.publish(CvBridge().cv2_to_imgmsg(mask, 'mono8'))
    if len(contours) != 0:
        cnt = max(contours, key=cv2.contourArea)
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > 30:
            return True
    return False

# this function normalizes the victor
def normalize(x, y):
    return x / math.sqrt(x ** 2 + y ** 2), y / math.sqrt(x ** 2 + y ** 2)

# callback function, which changes the coordinates of the center of the bigest contour
# this func is needed to "preciseLanding" func
def getCenterOfContour_callback(data):
    global cx
    global cy
    im = CvBridge().imgmsg_to_cv2(data, 'bgr8')
    out = im.copy()
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    kernal = np.ones((3, 3), np.uint8)
    if IS_GAZEBO:
        mask = cv2.morphologyEx(cv2.inRange(hsv, np.array(
            [55, 110, 110]), np.array([65, 255, 255])), cv2.MORPH_OPEN, kernal)
    else:
        mask = cv2.morphologyEx(cv2.inRange(hsv, np.array(
            [60, 50, 70]), np.array([85, 255, 255])), cv2.MORPH_OPEN, kernal)

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    if len(contours) != 0:
        cnt = max(contours, key=cv2.contourArea)
        [x, y, w, h] = cv2.boundingRect(cnt)  # getting the the coordinates of the bigest contour 
        cx = x + w // 2
        cy = y + h // 2

    out = cv2.arrowedLine(out, (320 // 2, 240 // 2), (cx, cy), (0, 0, 255), 4)
    out_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    vis = np.concatenate((out, out_mask), axis=1)

    debug.publish(CvBridge().cv2_to_imgmsg(vis, 'bgr8'))

# the function that implements the precision landing algorithm
# snapping to the center of the contour is underway
def preciseLanding(cur_point):  
    global cx
    global cy
    z = 1.8
    image_sub = rospy.Subscriber(
        'main_camera/image_raw_throttled', Image, getCenterOfContour_callback, queue_size=1)
    rospy.sleep(1)

    x0, y0 = 320 // 2, 240 // 2

    while math.sqrt((x0 - cx) ** 2 + (y0 - cy) ** 2) > 5:

        if math.sqrt((x0 - cx) ** 2 + (y0 - cy) ** 2) < 50:
            z = 1.7
        if math.sqrt((x0 - cx) ** 2 + (y0 - cy) ** 2) < 20:
            z = 1.55

        telem = get_telemetry(frame_id='aruco_map')

        if math.sqrt((telem.x - cur_point[0]) ** 2 + (telem.y - cur_point[1]) ** 2) > 1:
            navigate_wait(x=cur_point[0], y=cur_point[1], z=1.8, tolerance=0.2)
            print('Return for landing')
            continue

        dx, dy = normalize(cx - x0, cy - y0)  # get final motion vector 
        dx /= 15  # limit the speed
        dy /= 15
        dy = -dy  # the y-axis of the frame is directed in the opposite direction of the y-axis of the marker map
        # z -= 0.03
        set_position(x=telem.x + dx, y=telem.y + dy, z=z,
                     yaw=math.radians(90), frame_id='aruco_map')
        rospy.sleep(0.1)

    image_sub.unregister()
    land()
    rospy.sleep(1)
    arming(False)
    cx, cy = 0, 0

# final landing in (x=0, y=0)
def finalLanding():
    landX, landY = 0, 0
    navigate_wait(x=landX, y=landY, z=1.5, speed=0.6,
                  frame_id='aruco_map', tolerance=0.10)
    rospy.sleep(2)
    navigate(x=landX, y=landY, z=0.8, speed=0.3,
             yaw=float('nan'), frame_id='aruco_map')
    rospy.sleep(4)

    land()
    rospy.sleep(3)
    arming(False)


'''PART_ONE'''
# takeoff
navigate(z=1.5, frame_id='body', auto_arm=True)
rospy.sleep(3)
set_effect(r=0, g=0, b=0)
navigate_wait(z=1.8)
# flight
navigate_wait(x=0.9, y=4.5, z=1.8)
navigate_wait(x=0.8, y=4.5, z=1.4)
rospy.sleep(3)

detectCargos(rospy.wait_for_message(  # detecting cargos
    'main_camera/image_raw_throttled', Image), 'left')

navigate_wait(x=2.8, y=4.5, z=1.4)
rospy.sleep(3)

detectCargos(rospy.wait_for_message(  # detecting cargos
    'main_camera/image_raw_throttled', Image), 'right')

navigate_wait(x=2.8, y=4.5, z=1.8)

# creating the firs report
balance = 0
for col in cargos_colors_quantity:
    balance += cargos_colors_quantity[col]
print('Balance %i cargo' % balance)
for i in range(len(cargos_type_num)):
    for t in cargos_type_num:
        if cargos_type_num[t] == i:
            print('Type %i: %i cargo' % (i, cargos_colors_quantity[t]))
print('')

'''PART_TWO'''

#creating an array of possible locations for dronepoints
possible_locations = []
for i in range(4):
    for j in range(5):
        if not i % 2:
            possible_locations.append([j * 0.9, (3 - i) * 0.9])
        else:
            possible_locations.append([(4 - j) * 0.9, (3 - i) * 0.9])
detectedDrPoints = []

# flight
drPointCount = 0
cx, cy = 0, 0
for point in possible_locations:
    navigate_wait(x=point[0], y=point[1], z=1.8, tolerance=0.2)

    if isDronePointHere():  #detecting the dronepoints
        drPointCount += 1
        navigate_wait(x=point[0], y=point[1], z=2)
        rospy.sleep(3)
        cur_num = detectDigits()  # get number on the dronepoint

        for d in digit_rec:  # if cur_num != (0, 1, 2, 3)
            for i in digit_rec[d]:
                if i == cur_num:
                    cur_num = d

        print('Detected %i' % cur_num)

        for t in cargos_type_num:
            if cargos_type_num[t] == cur_num:
                set_effect('fade', *cargos_led_colors[t])  # led indications
        rospy.sleep(6)

        '''
        land()
        rospy.sleep(3)
        arming(False)
        '''

        preciseLanding(point)  # landing on the dronepoint

        for t in cargos_type_num:
            if cargos_type_num[t] == cur_num:
                print('D%i_delivered %s\n' % (cur_num, cargos_type[t]))
                detectedDrPoints.append([cur_num, cargos_colors_quantity[t]])
                break
        rospy.sleep(4)

        print('takeoff')
        navigate_wait(z=1.5, yaw=float('nan'), auto_arm=True,
                      frame_id='body', speed=1.5, tolerance=0.3)
        print('')
        set_effect(effect='fade', r=0, g=0, b=0)

    if drPointCount >= 2:
        break

# final landing
finalLanding()

# creating the final report
print('D%i_delivered to %i cargo' %
      (detectedDrPoints[0][0], detectedDrPoints[0][1]))
print('D%i_delivered to %i cargo' %
      (detectedDrPoints[1][0], detectedDrPoints[1][1]))
print('Balance: %i' %
      (balance - (detectedDrPoints[0][1] + detectedDrPoints[1][1])))