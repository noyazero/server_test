import cv2
import time
import asyncio
import datetime
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURACIÓN ---
CONFIG_MODELOS = {
    "fuego": { "archivo": "fuego-seg.pt", "defaults": {"fire": 0.50, "smoke": 0.50} },
    "facial": { "archivo": "yolov8n-face.pt", "defaults": {"face": 0.60} },
    "patente": { "archivo": "best_plate.pt", "defaults": {"license_plate": 0.40} },
    "perimetro": { "archivo": "yolov8n.pt", "defaults": {"person": 0.50} }
}

sistema = {
    "modelo_activo": None,
    "id_modelo": "ninguno",
    "socket": None,
    "umbrales": {}
}

# --- ENDPOINTS ---
@app.post("/set_model/{nombre}")
async def cambiar_modelo(nombre: str):
    sistema["modelo_activo"] = None
    sistema["id_modelo"] = "cargando..."
    
    if nombre not in CONFIG_MODELOS: return {"status": "error"}
    
    cfg = CONFIG_MODELOS[nombre]
    if not os.path.exists(cfg["archivo"]):
        sistema["id_modelo"] = "error_archivo"
        return {"status": "error", "msg": "Archivo no encontrado"}

    try:
        sistema["modelo_activo"] = YOLO(cfg["archivo"])
        sistema["id_modelo"] = nombre
        if nombre not in sistema["umbrales"]:
            sistema["umbrales"] = cfg["defaults"].copy()
        return {"status": "ok", "configuracion": sistema["umbrales"]}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

@app.post("/set_class_conf/{clase}/{valor}")
async def ajustar_clase(clase: str, valor: float):
    sistema["umbrales"][clase] = valor / 100.0
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sistema["socket"] = websocket
    try:
        while True: await websocket.receive_text()
    except: sistema["socket"] = None

# --- PROCESAMIENTO (SOLO DETECCIÓN) ---
def procesar_frame(frame, resultados):
    anotado = frame.copy()
    alertas = []
    
    det = resultados[0].boxes
    if len(det) > 0:
        for box in det:
            try:
                clase_id = int(box.cls[0])
                conf = float(box.conf[0])
                nombre_clase = resultados[0].names[clase_id] # ej: "fire", "smoke"
            except: continue

            # 1. FILTRO DE SENSIBILIDAD (SLIDER)
            umbral = sistema["umbrales"].get(nombre_clase, 0.5)
            if conf < umbral: continue 

            # 2. DIBUJAR (Sin calcular tiempo aquí)
            color = (0, 255, 0) # Siempre verde en el video raw, el frontend decidirá la alerta
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(anotado, (x1, y1), (x2, y2), color, 2)
            cv2.putText(anotado, f"{nombre_clase.upper()} {int(conf*100)}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 3. ENVIAR DATA CRUDA AL FRONT
            alertas.append({
                "clase_raw": nombre_clase, # ESTO ES LO IMPORTANTE PARA EL FRONT
                "msg": f"Detectado: {nombre_clase.upper()}",
                "confianza": int(conf*100)
            })

    return anotado, alertas

def generar_frames():
    cap = cv2.VideoCapture(0)
    ultimo_ws = 0
    ws_cooldown = 0.2 # Más rápido para que el frontend cuente mejor

    while True:
        success, frame = cap.read()
        if not success: break

        if sistema["modelo_activo"] is not None:
            # Detección laxa
            resultados = sistema["modelo_activo"].predict(frame, conf=0.01, verbose=False)
            frame_final, alertas = procesar_frame(frame, resultados)
            
            now = time.time()
            if len(alertas) > 0 and (now - ultimo_ws > ws_cooldown) and sistema["socket"]:
                # Enviamos TODAS las alertas detectadas en el frame, no solo una
                # Pero para simplificar el JSON enviamos la primera o iteramos en el front
                alerta = alertas[0]
                
                payload = {
                    "tiempo": datetime.datetime.now().strftime("%H:%M:%S"),
                    "modelo": sistema["id_modelo"],
                    "clase_raw": alerta["clase_raw"], # "fire", "smoke"
                    "mensaje": alerta["msg"],
                    "confianza": alerta["confianza"]
                }
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(sistema["socket"].send_json(payload))
                    ultimo_ws = now
                except: pass
        else:
            frame_final = frame
            msg = "ESPERANDO MODELO"
            if sistema["id_modelo"] == "error_archivo": msg = "ARCHIVO NO ENCONTRADO"
            cv2.putText(frame_final, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        ret, buffer = cv2.imencode('.jpg', frame_final)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generar_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)