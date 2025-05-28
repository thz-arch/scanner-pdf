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
    # Aumenta contraste com CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # Binarização fixa para destacar folha branca
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    # Inverter se fundo for escuro
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    # Fechamento morfológico (kernel maior para unir bordas)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.dilate(closed, kernel, iterations=2)

    # Borda com Canny mais sensível
    edged = cv2.Canny(closed, 30, 100)

    # Salva imagens intermediárias para debug
    cv2.imwrite("debug_gray.jpg", gray)
    cv2.imwrite("debug_thresh.jpg", thresh)
    cv2.imwrite("debug_closed.jpg", closed)
    cv2.imwrite("debug_edged.jpg", edged)

    # Contornos
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Exception("Nenhum contorno encontrado")

    # Escolhe o contorno de maior área
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    img_area = orig.shape[0] * orig.shape[1]
    # Se o maior contorno for menor que 30% da imagem, faz fallback
    if area < 0.3 * img_area:
        h, w = orig.shape[:2]
        box = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype="int")
    else:
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect).astype("int")

    # Desenha para debug
    dbg = orig.copy()
    cv2.drawContours(dbg, [box], -1, (0, 255, 0), 3)
    cv2.imwrite("debug_minAreaRect.jpg", dbg)

    # Usa 'box' como doc_cnt
    doc_cnt = box.reshape(4,1,2)

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

    try:
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
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise Exception(f"Erro no processamento: {str(e)}\nTraceback: {tb}")

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
