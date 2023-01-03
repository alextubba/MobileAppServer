import base64
from PIL import Image

from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def home():
    
    return "Hello, World!"
    
if __name__ == "__main__":
    app.run(debug=True)
    
    decoded_image = base64.b64decode(request.args.get("data"))
    print(type(decoded_image))

    image = Image.frombytes('RGBA', (128,128), decoded_image)
    image.show()
    print("Got message")