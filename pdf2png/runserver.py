from flask import Flask, request
from flask_cors import CORS
from copy import deepcopy

import fitz
import time
import os

app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route('/transformflie', methods=["POST"])
def pdf2png():
    zoom_x = 7
    zoom_y = 7
    # 获取普通参数
    print(f"request.values: {request.values}")
    print(f"request.values.get('name''): {request.values.get('name')}")

    # 获取文件类型的参数
    print(f"request.files: {request.files}")
    print(f"request.files.get('file_name'): {request.files.get('file_name')}")
    # 获取上传文件的二进制流
    file = request.files.get('file_name').read()
    pdf = fitz.open(stream=file, filetype='pdf')

    return_imgs = b''
    for pg in range(0, pdf.pageCount):
        page = pdf[pg]
        # 设置缩放和旋转系数
        trans = fitz.Matrix(zoom_x, zoom_y).prerotate(0)
        pm = page.getPixmap(matrix=trans, alpha=False)
        name = str(time.time()) + '.png'
        pm.save(name)
        with open(name, 'rb') as F:
            img = F.read()
            return_img = deepcopy(img)
        if pg == 0:
            return_imgs += return_img
        else:
            return_imgs += b'fenge' + return_img
        try:
            os.remove(name)
        except:
            print('删除文件失败')
            pass
    pdf.close()

    return return_imgs


if __name__ == "__main__":
    # app.run(debug=True, host='0.0.0.0', port=5016)
    app.debug = True
    app.run(host='0.0.0.0', port=5002)
    # app.debug = True
    # app.run(port=5001)
