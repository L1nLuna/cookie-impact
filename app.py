import pyautogui
import cv2
import numpy as np
from ultralytics import YOLO
import time
import tkinter as tk
from tkinter import ttk
import threading

model = YOLO("runs/detect/cookie_derector-2/weights/best.onnx")

is_running = False
conf_threshold = 0.7


def find_and_click():
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    results = model(screenshot_np, conf=0.1)

    clicked_any = False
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf.item()

            if confidence > conf_threshold:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                pyautogui.moveTo(center_x, center_y, duration=0.1)
                pyautogui.click()
                print(f'Клик: ({center_x}, {center_y}), Conf: {confidence:.2f}')
                clicked_any = True
                time.sleep(0.3)
    return clicked_any


def bot_loop():
    global is_running
    while is_running:
        try:
            if find_and_click():
                time.sleep(2)
            else:
                time.sleep(0.5)
        except Exception as e:
            print(f"Ошибка: {e}")
            time.sleep(1)


def start_bot(event=None):
    global is_running
    if not is_running:
        is_running = True
        print("▶ Бот запущен")
        threading.Thread(target=bot_loop, daemon=True).start()


def stop_bot(event=None):
    global is_running
    if is_running:
        is_running = False
        print("⏹ Бот остановлен")


root = tk.Tk()
root.title("Настройки кликера")
root.geometry("320x260")
root.attributes("-topmost", True)

root.bind("<space>", stop_bot) 
root.bind("<Return>", start_bot)

tk.Label(root, text="Порог уверенности (Confidence):").pack(pady=5)
conf_label = tk.Label(root, text=f"{conf_threshold:.2f}")
conf_label.pack()

def update_conf(val):
    global conf_threshold
    conf_threshold = float(val)
    conf_label.config(text=f"{conf_threshold:.2f}")

slider = ttk.Scale(root, from_=0.1, to=1.0, orient="horizontal", command=update_conf)
slider.set(conf_threshold)
slider.pack(pady=5, padx=20, fill="x")

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Старт", command=start_bot, bg="green", fg="white", width=10).pack(side="left", padx=5)
tk.Button(btn_frame, text="Стоп", command=stop_bot, bg="red", fg="white", width=10).pack(side="left", padx=5)

tk.Label(root, text="SPACE — стоп\n🟢 ENTER — старт", fg="gray").pack(pady=5)

def on_close():
    stop_bot()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()