import cv2
import numpy as np
import tempfile
from fpdf import FPDF
from PIL import Image
from flask import Flask, request, send_file, jsonify
import os
import json
import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
import google.auth
import base64

app = Flask(__name__)

def get_documentai_vertices(image_path, service_account_json=None):
    """
    Envia a imagem para o Google Document AI e retorna os vértices do documento, se encontrados.
    """
    endpoint = "https://us-documentai.googleapis.com/v1/projects/803691180758/locations/us/processors/97093285b878f134:process"
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(GoogleAuthRequest())
    token = creds.token
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    # Lê e codifica a imagem em base64
    with open(image_path, "rb") as f:
        img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {
        "rawDocument": {
            "content": img_b64,
            "mimeType": "image/jpeg"
        }
    }
    response = requests.post(endpoint, headers=headers, json=payload)
    if response.status_code != 200:
        print('Document AI erro:', response.status_code, response.text)
        return None
    data = response.json()
    # Loga a resposta completa para debug se não encontrar vértices
    try:
        vertices = data["document"]["pages"][0]["detectedDocument"]["layout"]["boundingPoly"]["normalizedVertices"]
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        points = [[int(v["x"]*w), int(v["y"]*h)] for v in vertices]
        if len(points) == 4:
            return np.array(points, dtype="float32")
    except Exception:
        print('Resposta completa do Document AI para debug:', json.dumps(data, indent=2, ensure_ascii=False))
    return None

def process_scan(image_path):
    # Ordenar e transformar
    def order_points(pts):
        pts = np.array(pts).reshape(4, 2)
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

    # Tenta ler vértices de um arquivo JSON (gerado por IA)
    vertices_path = image_path.replace('.jpg', '.json')
    if os.path.exists(vertices_path):
        with open(vertices_path, 'r') as f:
            vertices = np.array(json.load(f), dtype='float32')
        image = cv2.imread(image_path)
        orig = image.copy()
        warped_color = four_point_transform(orig, vertices)
        # Pós-processamento: nitidez e binarização forte
        kernel_sharp = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        warped_color = cv2.filter2D(warped_color, -1, kernel_sharp)
        warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
        _, warped_bin = cv2.threshold(warped_gray, 180, 255, cv2.THRESH_BINARY)
        warped_final = cv2.cvtColor(warped_bin, cv2.COLOR_GRAY2BGR)

        # Salva imagem final para PDF
        temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temp_img = temp_file.name.replace('.pdf', '.jpg')
        cv2.imwrite(temp_img, warped_final)
        pdf = FPDF()
        pdf.add_page()
        pdf.image(temp_img, x=10, y=10, w=190)
        pdf.output(temp_file.name)
        return temp_file.name
    image = cv2.imread(image_path)
    orig = image.copy()
    
    # Pré-processamento
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Suaviza sombras
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Binarização adaptativa
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 15)
    # Inverter se fundo for escuro
    if np.mean(thresh) < 127:
        thresh = cv2.bitwise_not(thresh)

    # Fechamento morfológico (kernel menor)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.dilate(closed, kernel, iterations=2)

    # Borda
    edged = cv2.Canny(closed, 30, 100)

    # Salva imagens intermediárias para debug
    cv2.imwrite("debug_gray.jpg", gray)
    cv2.imwrite("debug_blur.jpg", blur)
    cv2.imwrite("debug_thresh.jpg", thresh)
    cv2.imwrite("debug_closed.jpg", closed)
    cv2.imwrite("debug_edged.jpg", edged)

    # Contornos
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: tenta detectar bordas com Document AI
        vertices = get_documentai_vertices(image_path)
        if vertices is not None:
            warped_color = four_point_transform(orig, vertices)
            # Pós-processamento: nitidez e binarização forte
            kernel_sharp = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
            warped_color = cv2.filter2D(warped_color, -1, kernel_sharp)
            warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
            _, warped_bin = cv2.threshold(warped_gray, 180, 255, cv2.THRESH_BINARY)
            warped_final = cv2.cvtColor(warped_bin, cv2.COLOR_GRAY2BGR)
            temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            temp_img = temp_file.name.replace('.pdf', '.jpg')
            cv2.imwrite(temp_img, warped_final)
            pdf = FPDF()
            pdf.add_page()
            pdf.image(temp_img, x=10, y=10, w=190)
            pdf.output(temp_file.name)
            return temp_file.name
        else:
            raise Exception("Nenhum contorno encontrado e IA também não detectou bordas.")

    # Tenta encontrar o maior contorno com 4 lados (folha)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    doc_cnt = None
    img_area = orig.shape[0] * orig.shape[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.4 * img_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            # Checa proporção do retângulo
            rect = cv2.minAreaRect(approx)
            (w, h) = rect[1]
            if w == 0 or h == 0:
                continue
            aspect = max(w, h) / min(w, h)
            if 1.2 < aspect < 1.7:
                doc_cnt = approx
                break
    # Se não encontrar, usa a imagem inteira
    if doc_cnt is None or len(doc_cnt) != 4:
        h, w = orig.shape[:2]
        doc_cnt = np.array([[[0,0]], [[w-1,0]], [[w-1,h-1]], [[0,h-1]]], dtype="int")

    # Debug visual
    dbg = orig.copy()
    cv2.drawContours(dbg, [doc_cnt], -1, (0, 255, 0), 3)
    cv2.imwrite("debug_minAreaRect.jpg", dbg)

    # Usa 'box' como doc_cnt
    doc_cnt = doc_cnt.reshape(4,1,2)

    # Debug opcional
    debug_img = orig.copy()
    cv2.drawContours(debug_img, [doc_cnt], -1, (0, 255, 0), 3)
    cv2.imwrite("detected_contour.jpg", debug_img)

    try:
        warped_color = four_point_transform(orig, doc_cnt)

        # Pós-processamento: nitidez e binarização forte
        kernel_sharp = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        warped_color = cv2.filter2D(warped_color, -1, kernel_sharp)
        warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
        _, warped_bin = cv2.threshold(warped_gray, 180, 255, cv2.THRESH_BINARY)
        warped_final = cv2.cvtColor(warped_bin, cv2.COLOR_GRAY2BGR)

        # Salva imagem final para PDF
        temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temp_img = temp_file.name.replace('.pdf', '.jpg')
        cv2.imwrite(temp_img, warped_final)
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
