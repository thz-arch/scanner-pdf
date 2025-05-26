from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import tempfile
from fpdf import FPDF
from PIL import Image

app = Flask(__name__)

def process_scan(image_path):
    # Lê imagem colorida
    image = cv2.imread(image_path)
    orig = image.copy()

    # Converte para tons de cinza apenas para detecção de contornos
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blur, 30, 100)

    # Encontra e ordena contornos
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Busca contorno retangular de maior área
    doc_cnt = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < 2000:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx
            break

    if doc_cnt is None:
        raise Exception("Documento não detectado")

    # Ordena pontos do retângulo
    def order_points(pts):
        pts = pts.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    # Função de perspectiva
    def four_point_transform(image, pts):
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    # Aplica transformação e mantém cor original
    warped_color = four_point_transform(orig, doc_cnt)

    # Salva imagem colorida extraída temporariamente
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_img = temp_file.name.replace('.pdf', '.jpg')
    cv2.imwrite(temp_img, warped_color)

    # Gera PDF com a imagem colorida
    pdf = FPDF()
    pdf.add_page()
    pdf.image(temp_img, x=10, y=10, w=190)
    pdf.output(temp_file.name)

    return temp_file.name

@app.route("/scan", methods=["POST"])
def scan():
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_input:
        file.save(temp_input.name)
        try:
            result_path = process_scan(temp_input.name)
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return send_file(
        result_path,
        mimetype='application/pdf',
        as_attachment=True,
        download_name='documento.pdf'
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
