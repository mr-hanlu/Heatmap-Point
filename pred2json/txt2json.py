# 将标签写为json格式

class Txt2Json():
    def __init__(self, clas_names):
        self.clas_names = clas_names

        self.out0 = '''{
        "version": "4.5.7",
        "flags": {},
        "shapes": ['''

        self.out1 = '''
          {
            "label": "%(class)s",
            "points": [
              [
                %(point_x)d,
                %(point_y)d
              ]
            ],
            "group_id": null,
            "shape_type": "point",
            "flags": {}
          },'''

        self.out2 = '''
        ],
        "imagePath": "%(img_name)s",
        "imageData": null,
        "imageHeight": %(imageHeight)d,
        "imageWidth": %(imageWidth)d
        }
        '''

    def save_json(self, img_name, json_path, imageWidth, imageHeight, points):
        source = {}
        label = {}
        fjson = open(json_path, 'w')

        fjson.write(self.out0)

        self.out1_ = self.out1
        for i in range(points.shape[0]):
            label['class'] = self.clas_names[i]

            label['point_x'] = int(float(points[i][0]))
            label['point_y'] = int(float(points[i][1]))

            if i == (points.shape[0] - 1):  # 删除最后一行多余的','号
                self.out1_ = self.out1_[:-1]
            fjson.write(self.out1_ % label)

        source['img_name'] = img_name
        source['imageHeight'] = imageHeight
        source['imageWidth'] = imageWidth
        fjson.write(self.out2 % source)


if __name__ == '__main__':
    txt2json = Txt2Json([0])
    print(txt2json.out0+txt2json.out1+txt2json.out2)