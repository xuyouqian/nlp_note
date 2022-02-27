import requests
from requests_toolbelt import MultipartEncoder

url = 'http://127.0.0.1:5002/transformflie'
# 用MultipartEncoder函数对参数进行编码
# 如果要POST文件，需要用元组上传文件信息
# 元组格式：(文件名, 文件二进制流, "application/octet-stream")

data = MultipartEncoder({
    "name": "upload test",
    "file_name": ("image", open(r"01_安装.pdf", "rb").read(), "application/octet-stream")
})

# 指定POST参数的编码格式
headers = {
    "Content-Type": data.content_type
}

# 发送请求
r = requests.post(url=url, headers=headers, data=data)
fenge = r.content.split(b'fenge')

for index, img in enumerate(fenge):
    with open(str(index) + '.png', 'wb') as F:
        F.write(img)
