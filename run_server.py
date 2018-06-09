# Copyright (c) Alex Ellis 2017. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

from flask import Flask, request, jsonify
from handler import handle

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def main_route():
    return handle(request.get_data())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
