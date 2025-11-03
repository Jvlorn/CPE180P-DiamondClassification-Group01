# client.py - Flet client for laptop
import flet as ft
import requests
import io
from PIL import Image
import base64
import json

class DiamondClassifierClient:
    def __init__(self, server_url="http://192.168.1.100:8000"):
        self.server_url = server_url
        self.is_connected = False
        
    def test_connection(self):
        """Test connection to server"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.is_connected = data.get("model_loaded", False)
                return True, "Connected to server - Model ready" if self.is_connected else "Connected but model not loaded"
            return False, f"Server error: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"Cannot connect to server: {e}"
    
    def predict_image(self, image_path):
        """Send image to server for prediction"""
        try:
            with open(image_path, "rb") as f:
                files = {"file": (image_path, f, "image/jpeg")}
                response = requests.post(
                    f"{self.server_url}/predict", 
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, f"Prediction failed: {response.text}"
                
        except Exception as e:
            return False, f"Error: {e}"

def main(page: ft.Page):
    page.title = "Diamond Quality Classifier"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    page.scroll = "adaptive"
    
    client = DiamondClassifierClient()
    
    # UI Components
    server_url_field = ft.TextField(
        label="Server URL",
        value="http://192.168.1.100:8000",
        width=400,
        hint_text="http://raspberry-pi-ip:8000"
    )
    
    connection_status = ft.Text("Not connected", color="red")
    selected_files = ft.Text("No file selected", style=ft.TextThemeStyle.BODY_MEDIUM)
    result_display = ft.Column(scroll="adaptive")
    image_preview = ft.Image(width=300, height=300, fit=ft.ImageFit.CONTAIN)
    
    def update_connection_status(message, is_success):
        connection_status.value = message
        connection_status.color = "green" if is_success else "red"
        page.update()
    
    def connect_to_server(e):
        client.server_url = server_url_field.value
        success, message = client.test_connection()
        update_connection_status(message, success)
    
    def pick_files_result(e: ft.FilePickerResultEvent):
        if e.files:
            selected_files.value = f"Selected: {e.files[0].name}"
            # Display image preview
            if e.files[0].path:
                image_preview.src = e.files[0].path
            page.update()
        else:
            selected_files.value = "No file selected"
            page.update()
    
    def classify_image(e):
        if not client.is_connected:
            update_connection_status("Not connected to server", False)
            return
            
        if not file_picker.result or not file_picker.result.files:
            update_connection_status("Please select an image first", False)
            return
            
        file_path = file_picker.result.files[0].path
        if not file_path:
            update_connection_status("Invalid file path", False)
            return
        
        # Show loading
        result_display.controls.clear()
        result_display.controls.append(
            ft.ProgressRing()
        )
        page.update()
        
        # Send to server
        success, result = client.predict_image(file_path)
        
        result_display.controls.clear()
        
        if success:
            display_prediction_results(result)
        else:
            result_display.controls.append(
                ft.Text(f"❌ Error: {result}", color="red")
            )
        
        page.update()
    
    def display_prediction_results(result):
        """Display prediction results in a nice format"""
        predictions = result.get("predictions", {})
        
        # Overall confidence
        avg_confidence = sum(pred['confidence'] for pred in predictions.values()) / len(predictions)
        
        result_display.controls.append(
            ft.Container(
                content=ft.Column([
                    ft.Text("🔍 Classification Results", 
                           size=20, weight=ft.FontWeight.BOLD, color="blue"),
                    ft.Text(f"Overall Confidence: {avg_confidence:.1%}", 
                           size=16, color="green"),
                ]),
                padding=10,
                bgcolor=ft.Colors.BLUE_50,
                border_radius=10,
                margin=5
            )
        )
        
        # Individual predictions
        for attribute, pred in predictions.items():
            confidence_color = "green" if pred['confidence'] > 0.7 else "orange" if pred['confidence'] > 0.5 else "red"
            
            result_display.controls.append(
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text(attribute.upper(), 
                                   weight=ft.FontWeight.BOLD, size=16),
                            ft.Text(f"Quality: {pred['label']}", size=14),
                            ft.Text(f"Confidence: {pred['confidence']:.1%}", 
                                   color=confidence_color, size=14),
                            # Probability distribution
                            ft.Container(
                                content=ft.Column([
                                    ft.Text("Details:", size=12, weight=ft.FontWeight.BOLD),
                                    *[ft.Text(f"  {label}: {prob:.1%}", size=11) 
                                      for label, prob in pred['all_probabilities'].items()]
                                ]),
                                margin=ft.margin.only(top=5)
                            )
                        ]),
                        padding=15,
                    ),
                    margin=5
                )
            )
    
    # File picker
    file_picker = ft.FilePicker(on_result=pick_files_result)
    page.overlay.append(file_picker)
    
    # Build UI
    page.add(
        ft.Column([
            ft.Text("💎 Diamond Quality Classifier", 
                   size=24, weight=ft.FontWeight.BOLD, color="blue"),
            
            ft.Row([
                server_url_field,
                ft.ElevatedButton("Connect", on_click=connect_to_server),
            ], alignment=ft.MainAxisAlignment.START),
            
            connection_status,
            
            ft.Divider(),
            
            ft.Text("Select Diamond Image:", size=16, weight=ft.FontWeight.BOLD),
            
            ft.Row([
                ft.ElevatedButton(
                    "📁 Choose Image",
                    icon=ft.Icons.FILE_UPLOAD,
                    on_click=lambda _: file_picker.pick_files(
                        allow_multiple=False,
                        file_type=ft.FilePickerFileType.IMAGE
                    )
                ),
                selected_files,
            ]),
            
            ft.Row([
                image_preview,
                ft.VerticalDivider(),
                ft.Container(
                    content=result_display,
                    width=400,
                    height=300,
                    padding=10,
                    border=ft.border.all(1, ft.Colors.GREY_300),
                    border_radius=10,
                )
            ]),
            
            ft.ElevatedButton(
                "🔍 Classify Diamond",
                on_click=classify_image,
                style=ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.BLUE,
                    padding=20
                ),
                icon=ft.Icons.SEARCH
            ),
            
            ft.Text("Instructions:", size=14, weight=ft.FontWeight.BOLD),
            ft.Text("1. Enter Raspberry Pi IP address (check with `hostname -I`)"),
            ft.Text("2. Click Connect to verify server connection"),
            ft.Text("3. Select a diamond image to classify"),
            ft.Text("4. Click Classify to analyze diamond quality"),
        ], scroll="adaptive")
    )

if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.FLET_APP)