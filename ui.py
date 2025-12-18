import flet as ft
import os
import asyncio
import traceback
import requests
import tempfile
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from dotenv import load_dotenv
from phone_agent import PhoneAgent
from phone_agent.agent import AgentConfig
from phone_agent.model import ModelConfig

# Load environment variables
load_dotenv()

def main(page: ft.Page):
    # Window settings: True Extreme Minimalism for macOS
    page.window_title_bar_hidden = True
    page.window_title_bar_buttons_hidden = True
    page.window_frameless = True  # More aggressive frameless
    
    # Transparency and No Title
    page.window_bgcolor = ft.Colors.TRANSPARENT
    page.bgcolor = ft.Colors.TRANSPARENT
    
    # Sizing and behavior
    page.window_width = 400
    page.window_height = 600
    page.window_resizable = False  # Fixed size for better frameless feel
    page.padding = 0
    page.spacing = 0

    # Recording State
    state = {"is_recording": False, "samplerate": 44100}
    recording_data = []

    # Initialize Agent
    model_config = ModelConfig(
        base_url=os.getenv("PHONE_AGENT_BASE_URL"),
        model_name=os.getenv("PHONE_AGENT_MODEL"),
        api_key=os.getenv("PHONE_AGENT_API_KEY"),
    )
    agent_config = AgentConfig(
        device_id=os.getenv("PHONE_AGENT_DEVICE_ID"),
        verbose=True,
    )
    agent = PhoneAgent(model_config=model_config, agent_config=agent_config)

    # Log Area
    log_area = ft.ListView(
        expand=True,
        spacing=10,
        padding=ft.padding.all(20),
        auto_scroll=True,
    )

    def add_log(message: str, color=ft.Colors.WHITE, weight=ft.FontWeight.NORMAL):
        print(f"UI LOG: {message}")
        log_area.controls.append(
            ft.Text(message, color=color, weight=weight)
        )
        page.update()

    def transcribe_audio(file_path):
        if not file_path: return ""
        url = "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"
        api_key = os.getenv("PHONE_AGENT_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            if file_path.startswith("file://"): file_path = file_path[7:]
            with open(file_path, "rb") as f:
                response = requests.post(url, headers=headers, files={"file": f}, data={"model": "glm-asr-2512"})
                response.raise_for_status()
                return response.json().get("text", "")
        except Exception as e:
            add_log(f"ASR Error: {str(e)}", color=ft.Colors.RED_400)
            return ""

    async def run_agent_task(task: str):
        add_log(f"Starting: {task}", color=ft.Colors.GREEN_400)
        try:
            agent.reset()
            add_log("Thinking...", color=ft.Colors.GREY_400)
            step_result = await asyncio.to_thread(agent.step, task)
            while True:
                add_log(f"💭 {step_result.thinking}", color=ft.Colors.GREY_400)
                add_log(f"🎯 {step_result.action}", color=ft.Colors.YELLOW_400)
                if step_result.finished: break
                step_result = await asyncio.to_thread(agent.step)
            add_log(f"✅ {step_result.message}", color=ft.Colors.GREEN_400, weight=ft.FontWeight.BOLD)
        except Exception as e:
            add_log(f"❌ Error: {str(e)}", color=ft.Colors.RED_400)

    def send_message(e):
        task = input_field.value.strip()
        if not task: return
        input_field.value = ""
        page.update()
        add_log(f"You: {task}", color=ft.Colors.BLUE_200, weight=ft.FontWeight.BOLD)
        page.run_task(run_agent_task, task)

    input_field = ft.TextField(
        hint_text="Enter task...",
        border_radius=20,
        filled=True,
        expand=True,
        border_color=ft.Colors.TRANSPARENT,
        bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.WHITE),
        on_submit=send_message,
        visible=True,
    )

    send_btn = ft.IconButton(
        icon=ft.Icons.SEND_ROUNDED,
        icon_color=ft.Colors.BLUE_400,
        on_click=send_message,
        visible=True,
    )

    voice_button = ft.IconButton(
        icon=ft.Icons.MIC_ROUNDED,
        icon_color=ft.Colors.WHITE,
        bgcolor=ft.Colors.BLUE_400,
        visible=False,
        scale=1.0,
        animate_scale=ft.Animation(600, ft.AnimationCurve.EASE_IN_OUT),
    )

    async def animate_recording():
        while state["is_recording"]:
            voice_button.scale = 1.3
            page.update()
            await asyncio.sleep(0.6)
            voice_button.scale = 1.0
            page.update()
            await asyncio.sleep(0.6)

    def audio_callback(indata, frames, time, status):
        recording_data.append(indata.copy())

    async def on_voice_click(e):
        if not state["is_recording"]:
            try:
                recording_data.clear()
                state["is_recording"] = True
                device_info = sd.query_devices(None, 'input')
                samplerate = int(device_info['default_samplerate'])
                state["samplerate"] = samplerate
                state["stream"] = sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback)
                state["stream"].start()
                voice_button.icon = ft.Icons.STOP_CIRCLE_ROUNDED
                voice_button.icon_color = ft.Colors.RED_400
                add_log("🎤 Listening...", color=ft.Colors.BLUE_200)
                asyncio.create_task(animate_recording())
            except Exception as ex:
                add_log(f"❌ Error: {str(ex)}", color=ft.Colors.RED_400)
                state["is_recording"] = False
        else:
            try:
                voice_button.disabled = True
                page.update()
                if "stream" in state:
                    state["stream"].stop()
                    state["stream"].close()
                state["is_recording"] = False
                voice_button.icon = ft.Icons.MIC_ROUNDED
                voice_button.icon_color = ft.Colors.WHITE
                voice_button.disabled = False
                if recording_data:
                    temp_path = os.path.join(tempfile.gettempdir(), "voice_input.wav")
                    audio_array = np.concatenate(recording_data, axis=0)
                    write(temp_path, state["samplerate"], audio_array)
                    add_log("⏳ Transcribing...", color=ft.Colors.GREY_400)
                    text = await asyncio.to_thread(transcribe_audio, temp_path)
                    if text:
                        input_field.value = text
                        add_log(f"✨ {text}", color=ft.Colors.BLUE_200)
                        send_message(None)
                    else:
                        add_log("❌ Failed.", color=ft.Colors.RED_400)
            except Exception as ex:
                add_log(f"❌ Error: {str(ex)}", color=ft.Colors.RED_400)
                state["is_recording"] = False
                voice_button.disabled = False
        page.update()

    voice_button.on_click = on_voice_click

    def toggle_input_mode(e):
        input_field.visible = not input_field.visible
        send_btn.visible = input_field.visible
        voice_button.visible = not input_field.visible
        toggle_btn.icon = ft.Icons.KEYBOARD_ROUNDED if voice_button.visible else ft.Icons.MIC_ROUNDED
        page.update()

    toggle_btn = ft.IconButton(
        icon=ft.Icons.MIC_ROUNDED,
        on_click=toggle_input_mode,
    )

    # Main Layout
    page.add(
        ft.WindowDragArea(
            content=ft.Container(
                width=400,
                height=600,
                bgcolor=ft.Colors.with_opacity(0.95, "#1E1E1E"),
                border_radius=24,
                padding=0,
                content=ft.Column(
                    [
                        # Logs Area
                        ft.Container(
                            content=log_area,
                            expand=True,
                        ),
                        
                        # Bottom Bar
                        ft.Container(
                            padding=ft.padding.only(left=20, right=20, bottom=30, top=10),
                            content=ft.Row(
                                [
                                    toggle_btn,
                                    ft.Stack(
                                        [
                                            ft.Row([input_field], alignment=ft.MainAxisAlignment.CENTER),
                                            ft.Row([voice_button], alignment=ft.MainAxisAlignment.CENTER),
                                        ],
                                        expand=True,
                                    ),
                                    send_btn,
                                ],
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                            ),
                        ),
                    ],
                    spacing=0,
                ),
            )
        )
    )
    
    # Extra force update for frameless
    page.window_center()
    page.update()

if __name__ == "__main__":
    ft.app(target=main)
