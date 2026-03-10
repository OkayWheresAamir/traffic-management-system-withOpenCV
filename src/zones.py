from shapely.geometry import Point, Polygon

# frame size = 640x360

far_zone = Polygon([(0,0),(640,0),(640,120),(0,120)])
queue_zone = Polygon([(0,120),(640,120),(640,240),(0,240)])
stop_zone = Polygon([(0,240),(640,240),(640,360),(0,360)])

def get_zone(x, y):

    p = Point(x, y)

    if far_zone.contains(p):
        return "far"

    if queue_zone.contains(p):
        return "queue"

    if stop_zone.contains(p):
        return "stop"

    return None