import math

class Rastreador:
    def __init__(self):
        self.centro_puntos = {}
        self.id_count = 1

    def rastreo(self, objectos):
        objectos_id = []
        for rect in objectos:
            x, y, w, h = rect
            cx = (x + w) / 2
            cy = (y + h) / 2

            objectos_det=False
            for id, pt in self.centro_puntos.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.centro_puntos[id] = (cx, cy)
                    print(self.centro_puntos[id])
                    objectos_id.append((x, y, w, h, id))
                    objectos_det=True
                    break
            if objectos_det is False:
                self.centro_puntos[self.id_count] = (cx, cy)
                objectos_id.append((self.id_count, x, y, w, h))
                self.id_count += self.id_count+1 #aumento el id

        new_center_point = {}
        for obj_bb_id in objectos_id:
            _, _, _, _, object_id = obj_bb_id
            new_center_point[object_id] = self.centro_puntos.get(object_id, (0, 0))

        self.centro_puntos = new_center_point.copy()
        return objectos_id
