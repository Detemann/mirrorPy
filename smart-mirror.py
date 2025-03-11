import tkinter as tk
from PIL import Image, ImageTk
import cv2
import speech_recognition as sr
from datetime import datetime
from dotenv import load_dotenv
import openai
import pygame
import os
import base64

load_dotenv()

openai.api_key = os.getenv('API_KEY')

CONFIG_PROMPT = """Você é um assistente virtual chamado Espelho, integrado a um espelho inteligente. 
Sua função é interagir com o usuário fornecendo informações úteis e respondendo perguntas de maneira educada e precisa.

Funções principais:
1. Cumprimentar o usuário com um 'Bom dia', 'Boa tarde' ou 'Boa noite', dependendo do horário.
2. Perguntar como o usuário está se sentindo e reagir de maneira empática.
3. Analisar a imagem do usuário e fornecer sugestões personalizadas, como:
   - Se a barba estiver grande, sugerir que ele faça a barba.
   - Se o cabelo estiver grande ou desarrumado, sugerir um corte ou que ele penteie.
   - Se o usuário parecer cansado ou doente, perguntar se ele está bem e sugerir cuidados básicos (descanso, hidratação, procurar um médico, etc.).
4. Responder perguntas sobre diversos temas, incluindo notícias, clima, agenda e saúde.
5. Caso você receba um prompt contendo a frase 'Rosto do usuário identificado pelo sistema', essa é uma imagem do usuário, e você deve analisá-la para fornecer recomendações apropriadas.

Seja sempre educado, prestativo e direto ao ponto."""

class SmartMirrorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Mirror")
        self.root.attributes('-fullscreen', True)
        self.photoSend = False

        self.canvas = tk.Canvas(root, bg="black")
        self.canvas.pack(fill="both", expand=True)

        self.video_item = self.canvas.create_image(0, 0, anchor="nw", image=None)

        self.time_text = self.canvas.create_text(50, 50, text="", fill="white", font=("Helvetica", 48), anchor="nw")
        self.face_text = self.canvas.create_text(50, 120, text="Nenhum rosto detectado", fill="white", font=("Helvetica", 36), anchor="nw")
        self.voice_text = self.canvas.create_text(50, 190, text="Aguardando comando de voz...", fill="white", font=("Helvetica", 36), anchor="nw")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Erro ao acessar a webcam")
        
        self.update_video()

        self.get_chatgpt_system_response(CONFIG_PROMPT)

        pygame.mixer.init()

        self.detect_face()

        self.update_time()

        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.start_listening()


        self.root.bind("<Escape>", lambda e: self.exit_app())

    def exit_app(self):
        pygame.quit()
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def update_video(self):
        """Captura frames da webcam e atualiza o fundo do Canvas."""
        ret, frame = self.cap.read()

        if ret:
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()
            if width < 1 or height < 1:
                width, height = 640, 480
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.itemconfig(self.video_item, image=self.photo)
        self.root.after(15, self.update_video)

    def update_time(self):
        """Atualiza o horário exibido no Canvas a cada 1 segundo."""
        now = datetime.now().strftime("%H:%M:%S")
        self.canvas.itemconfig(self.time_text, text=now)
        self.root.after(1000, self.update_time)
    
    def ttsAudio(self, message):
        response = openai.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=message,
            response_format="mp3"
        )

        response.stream_to_file("output.mp3")
        pygame.mixer.music.load('output.mp3')
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.music.stop()
        pygame.mixer.music.unload()

        os.remove('output.mp3')

    def detect_face(self):
        """Captura frames e realiza a detecção de rosto utilizando OpenCV."""
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                text = "Rosto detectado"
                cv2.imwrite('faceFrame.png', frame)
                if(not self.photoSend):
                    self.canvas.itemconfig(self.voice_text, text=f"Resposta: {self.get_chatgpt_response('Olá')}")
            else: 
                text = "Nenhum rosto detectado"
            self.canvas.itemconfig(self.face_text, text=text)
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def get_chatgpt_response(self, prompt):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )

            self.ttsAudio(response.choices[0].message.content.strip())
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            return f"Erro ao acessar o ChatGPT: {e}"
    
    def get_chatgpt_system_response_image(self):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": "Rosto do usuario identificado pelo sistema"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image('faceFrame.png')}"}}
                    ]}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            return f"Erro ao acessar o ChatGPT: {e}"
        
    def get_chatgpt_system_response(self, prompt):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            return f"Erro ao acessar o ChatGPT: {e}"

    def start_listening(self):
        """Inicia o reconhecimento de voz em background."""
        def callback(recognizer, audio):
            try:
                command = recognizer.recognize_google(audio, language='pt-BR')
                if "espelho" in command:
                    self.canvas.itemconfig(self.voice_text, text=f"Gerando resposta...")
                    response = self.get_chatgpt_response(command)
                    self.canvas.itemconfig(self.voice_text, text=f"Resposta: {response}")
            except sr.UnknownValueError:
                print("Não entendi o comando")
            except sr.RequestError as e:
                self.canvas.itemconfig(self.voice_text, text=f"Erro no serviço: {e}")

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        self.stop_listening = self.recognizer.listen_in_background(self.microphone, callback)

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartMirrorApp(root)
    root.mainloop()
