import cv2
import numpy as np
import tempfile
from fpdf import FPDF
from PIL import Image
from flask import Flask, request, send_file, jsonify

app = Flask(__name__)

def process_scan(image_path):
    image = cv2.imread(image_path)
    orig = image.copy()
    
    # Pré-processamento
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Binarização adaptativa para destacar a folha
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Inverter se fundo for escuro
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    # Fechamento morfológico
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.dilate(closed, kernel, iterations=2)

    # Borda
    edged = cv2.Canny(closed, 50, 150)

    # Contornos
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    def score_contour(cnt):
        area = cv2.contourArea(cnt)
        if area < 10000:
            return 0
        x, y, w, h = cv2.boundingRect(cnt)
        ar = float(w) / h if h else 0
        if not (0.5 < ar < 2.5):  # proporção da folha
            return 0
        return area

    candidates = sorted(contours, key=score_contour, reverse=True)
    doc_cnt = None

    for cnt in candidates:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx
            break

    if doc_cnt is None:
        print("Nenhuma borda clara encontrada, fallback para imagem inteira")
        h, w = gray.shape
        doc_cnt = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]])

    # Debug opcional
    debug_img = orig.copy()
    cv2.drawContours(debug_img, [doc_cnt], -1, (0, 255, 0), 3)
    cv2.imwrite("detected_contour.jpg", debug_img)

    # Ordenar e transformar
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

    warped_color = four_point_transform(orig, doc_cnt)

    # Gera PDF
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
