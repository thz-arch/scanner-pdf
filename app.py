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

    # Pré-processamento: reduz ruído mantendo bordas
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)

    # Aumenta contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blur)

    # Fechamento morfológico para remover pequenos espaços
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Detecta bordas com Canny adaptado
    minVal, maxVal = 50, 150
    edged = cv2.Canny(closed, minVal, maxVal)

    # Aproximação de contornos
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Função de pontuação: área e proporção mais flexível para comprovantes
    def score_contour(cnt):
        area = cv2.contourArea(cnt)
        if area < 1000:
            return 0
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            return 0
        x, y, w, h = cv2.boundingRect(approx)
        ar = float(w) / h if h else 0
        # Receipts podem ser verticais ou horizontais, proporção entre 0.5 e 3
        if not (0.5 < ar < 3.0):
            return 0
        return area

    # Escolhe melhor contorno
    candidates = sorted(contours, key=score_contour, reverse=True)
    doc_cnt = None
    if candidates and score_contour(candidates[0]) > 0:
        doc_cnt = cv2.approxPolyDP(candidates[0], 0.02 * cv2.arcLength(candidates[0], True), True)
    
    # Fallback: contorno da borda da imagem inteira
    if doc_cnt is None:
        h, w = gray.shape
        doc_cnt = np.array([[[0,0]], [[w,0]], [[w,h]], [[0,h]]])

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

    # Transforma perspectiva
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
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Aplica transformação e mantém cores originais
    warped_color = four_point_transform(orig, doc_cnt)

    # Salva PDF
    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temp_img = temp_file.name.replace('.pdf', '.jpg')
    cv2.imwrite(temp_img, warped_color)
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
        download_name='comprovante.pdf'
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
