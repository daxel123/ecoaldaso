import cv2
import base64
import numpy as np
import json
#quiero guardar en un archivo json llamaado recursos.json la siguiente estructura de datos



user_logo_path = "imagenes/user.png"  # Ruta a tu logo de usuario
with open(user_logo_path, "rb") as image_file:
    user_logo_str = base64.b64encode(image_file.read()).decode()

#Guardar el string en un archivo json
data_struc = {"name": "user_logo",
              "imagen": user_logo_str}

with open("recursos.json", "w") as file:
    json.dump(data_struc, file, indent=4)


home_image_path = "imagenes/contenedores.jpg"  # Ruta a tu imagen de inicio
with open(home_image_path, "rb") as image_file:
    home_image_str = base64.b64encode(image_file.read()).decode()

#Guardar el string en recursos.json
data_struc = {"name": "home_image",
              "imagen": home_image_str}

with open("recursos.json", "a") as file:
    json.dump(data_struc, file, indent=4)
    

