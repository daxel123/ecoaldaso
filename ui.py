import flet as ft
from flet import Page, Text, Container, Row, Column, DataTable, DataColumn, Image, ElevatedButton, Tabs, Tab, Divider, TextSpan
from detection import detect_objects, camera
from database import load_records_from_json
from charts import generate_historical_data, create_chart,generate_total_co2_per_material,plot_co2_by_material
from utils import update_record_list
import cv2
import base64
import numpy as np

def main_page(page: Page):
    page.window_width = 768
    page.window_height = 1024
    page.bgcolor = "#ffffff"

    user_logo_path = "imagenes/user.png"
    with open(user_logo_path, "rb") as image_file:
        user_logo_str = base64.b64encode(image_file.read()).decode()

    header = Container(
        content=Row(
            controls=[
                Text("Bienvenido a ECO Aldaso!", size=30, weight="bold")
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        ),
        padding=10
    )

    myresult = Column()
    record_list = DataTable(
        columns=[
            DataColumn(Text("Objeto")),
            DataColumn(Text("Confianza")),
            DataColumn(Text("Puntos")),
            DataColumn(Text("Fecha")),
        ],
        rows=[]
    )

    object_records = load_records_from_json()
    update_record_list(record_list, object_records, page)

    black_image = np.zeros((420, 420, 3), np.uint8)
    _, buffer = cv2.imencode('.jpg', black_image)
    black_image_str = base64.b64encode(buffer).decode()
    image_control = Image(src_base64=black_image_str)
    detection_text = Text("Detectado: None", size=25, weight="bold")

    instructions_label = Text("Instrucciónes: None", size=25, weight="bold")

    home_image_path = "imagenes/contenedores.jpg"
    with open(home_image_path, "rb") as image_file:
        home_image_str = base64.b64encode(image_file.read()).decode()

    def switch_tab(index):
        tab_control.selected_index = index
        page.update()

    intro_text = Text(
        spans=[
            TextSpan("Esta aplicación utiliza la tecnología YOLO para reconocer diferentes tipos de residuos e indicarte como debes reciclarlos. Puedes empezar por, "),
            TextSpan("aquí", on_click=lambda e: switch_tab(0)),
            TextSpan(", o ir a la sección de "),
            TextSpan("Detector", on_click=lambda e: switch_tab(1), style={"color": "blue", "underline": "true"}),
            TextSpan(" para iniciar la cámara y comenzar a detectar objetos. También puedes revisar los "),
            TextSpan("Registros", on_click=lambda e: switch_tab(2), style={"color": "blue", "underline": "true"}),
            TextSpan(" de detección anteriores o ver el "),
            TextSpan("Historial de Puntos", on_click=lambda e: switch_tab(3), style={"color": "blue", "underline": "true"}),
            TextSpan(" acumulados.")
        ],
        size=20
    )

    home_content = Column([
        header,
        intro_text,
        Image(src_base64=home_image_str, width=420, height=420)
    ])

    chart_image_str_CO= plot_co2_by_material(object_records)
    chart_image_2 = Image(src_base64=chart_image_str_CO)

    months, points = generate_historical_data(object_records)
    chart_image_str = create_chart(months, points)
    chart_image = Image(src_base64=chart_image_str)

    historical_content = Column([
        Text("Histórico de puntos", size=30, weight="bold"),
        chart_image,
        Text("Histórico de CO₂ no emitido", size=30, weight="bold"),
        chart_image_2
    ],scroll=ft.ScrollMode.ALWAYS)

    def on_camera_button_click(e):
        camera(image_control, page)

    def on_detect_button_click(e):
        detect_objects(instructions_label, detection_text, image_control, object_records, record_list, page)

    global tab_control
    tab_control = Tabs(
        selected_index=0,
        animation_duration=300,
        tabs=[
            Tab(text="Home", content=home_content),
            Tab(text="Detector", content=Column([
                Text("Detector de residuos", size=30, weight="bold"),
                Divider(),
                Row([
                    Column([
                        detection_text,
                        image_control,
                        
                    ]),
                    Container(
                        content=Column([
                            instructions_label
                        ]),
                        width=300,  # Set a fixed width for the container
                        padding=10,
                        bgcolor="#f0f0f0",  # Light grey background for better readability
                        border_radius=8,
                        
                    )
                ], alignment=ft.MainAxisAlignment.START),
                Row([
                    ElevatedButton("Habilitar cámara", bgcolor="blue", color="white", on_click=on_camera_button_click),
                    ElevatedButton("Captura el residuo", bgcolor="#00c900", color="white", on_click=on_detect_button_click),
                ])
            ])),
            Tab(text="Registros", content=Column([
                Text("Registro de objetos detectados", size=30, weight="bold"),
                record_list
            ], scroll=ft.ScrollMode.ALWAYS)),
            Tab(text="Datos históricos", content=Column([historical_content],scroll=ft.ScrollMode.ALWAYS)),
            Tab(icon=ft.Icon(ft.icons.PERSON_2_ROUNDED), content=Column([
                Text("Perfil del usuario", size=30, weight="bold"),
                Image(src_base64=user_logo_str, width=100, height=100)
            ]))
        ],
        expand=1
    )
    tab_control.divider_color = "#00c900"
    tab_control.divider_height = 5
    tab_control.indicator_color = "#00c900"
    tab_control.label_color = "#00c900"
    tab_control.overlay_color = "#00c900"

    page.add(tab_control)
