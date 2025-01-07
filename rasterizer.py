from cameraObj import camera
import numpy as np
from Scene_Obj import Scene, Triangle, Rectangle, RectanglePrisma
from rasterizing_algorithms import Bresenham_Algorithm, PrintTriangle, PrintTriangleWireframe
from numba import njit, jit
from numba import cuda
import timeit
import glfw
from OpenGL.GL import *
from OpenGL.GLUT import *
from PIL import Image

@njit
def print_image(image, depth_buffer, texture_array, changeBaseMatrix, vertexs, triangles, triangle_color, triangle_specular, triangle_reflectance, camera_pos, d, len_x, len_y, len_pixelX, len_pixelY):
 
    # \\\\ Screen Intersection: \\\\
    # Fem un canvi de base, ens situem a la base de la camara. Efectuem la translacio a la base de la camara i rotem els eixos:
    translated_vertexs = vertexs - camera_pos
    
    # Efectua el producte de matrius per cada fila
    new_base_vertexs = np.ascontiguousarray(np.empty_like(translated_vertexs))
    n = 0
    for vertex in translated_vertexs:
        new_base_vertexs[n] = np.dot(changeBaseMatrix, vertex)
        n += 1
    
    # w = z /d on d es la distancia respecte el pla de projeccio, i z es la coordenada del punt que volem projectar.
    # Suposem que estem al origen, i el pla de projeccio esta a z = d
    # Les coordenades en projeccio perspectiva son: x' = x / w ; y' = y / w; z' = d
    # Les coordenades fan que el pla de projeccio estigui al pla XY
    
    # El z_axis es super important, ja que al ser una coordenada perpendicular al pla de projeccio, ens diu la distancia de cada punt respecte el pla
    # Aixo ens permetra calcular correctament el Z-buffering. Un error que he fet era calcular el modul de distancia respecte el punt de projeccio
    # Aquest calcul no em servia per saber la profunditat de cada punt, ja que el que importa es la distancia respecte el pla de projeccio
    z_axis = new_base_vertexs[:,2]
    
    # Cada vertex te la seva w - parametre de projeccio, ens permet calcular la projeccio dels punts al pla de projeccio
    w = z_axis / d
   
    z_axis_plane = z_axis - d  # Distancia de cada punt respecte el pla de projeccio
    
    projected_vertexs = new_base_vertexs / w[:, np.newaxis]

    # Hem de representar-lo respecte el (0,0) de la pantalla, o sigui
    # el punt que esta a dalt a la esquerra per imatges. Un sistema de coordenades on X creix a la dreta i Y creix cap avall.
    # Recordem que el sistema de coordenades de la pantalla esta definit com X positiva a la esquerra i Y positiva cap amunt.
    
    # Coordenades de la pantalla
    pantalla_x = projected_vertexs[:,0] + len_x / 2
    pantalla_y = projected_vertexs[:,1] + len_y / 2
    
    # Mirem les coordenades del pixel al que li correspon aquest punt.
    pixel_coords_x = np.round(pantalla_x / len_pixelX).astype(np.int32)
    pixel_coords_y = np.round(pantalla_y / len_pixelY).astype(np.int32) # L'eix Y esta invertit en les imatges
    
    # Si t es negatiu vol dir que el vertex esta darrera la camera.
    # El parametre t es la distancia del centre de la camera al punt de interseccio amb la pantalla, ja que haviem fet unitari el vector director de la recta.
    # I aquest vector anava del centre de la camara cap al vertex. EL parametre t per tant ens dona una idea de la profunditat del vertex.
    
    for n in range(triangles.shape[0]):
        
        indexA = triangles[n,0]
        indexB = triangles[n,1]
        indexC = triangles[n,2]
        
        color = triangle_color[n]
        
        if z_axis_plane[indexA] > 0 or z_axis_plane[indexB] > 0 or z_axis_plane[indexC] > 0:
        # Pintem el triangle si almenys un dels seus vertexs esta davant el pla de la camera: 
            
            PrintTriangle(image,
                          depth_buffer,
                          texture_array,
                          pixel_coords_x[indexA], pixel_coords_y[indexA], 
                          pixel_coords_x[indexB], pixel_coords_y[indexB],
                          pixel_coords_x[indexC], pixel_coords_y[indexC], 
                          color, 
                          z_axis[indexA],
                          z_axis[indexB],
                          z_axis[indexC],
                          w[indexA],
                          w[indexB],
                          w[indexC])
            
    return image

# Defim l'escena:
scene = Scene()

# Definim una camara en la nostra escena:
camera_pos = 15*np.array([1,0,0])   
d = 8
pixels_x = 1024
pixels_y = 1024
len_x = 10
len_y = 10

Camera = camera(camera_pos, d, pixels_x, pixels_y, len_x, len_y)
Camera.followPoint([0,0,0])

# Coordenades [X, Y, Z]:
# scene.addInstance(RectanglePrisma([-1,1,0], [1,1,0], [-1,-1,0], [-1,1,5], [0,0,255], 0.2, 0))

# scene.addInstance(Triangle([-0.5,0,0],[1,5,0],[1,0,6], [0,0,250], 0.2, 0))
scene.addInstance(Triangle([0,0,0],[0,5,0],[0,0,6], [0,250,0], 0.2, 0))

# scene.addInstance(Triangle([0,0,0],[6,0,0],[0,5,0], [0,0,250], 0.2, 0))
# scene.addInstance(Triangle([0,0,0],[0,5,0],[0,0,6], [0,250,0], 0.2, 0))

# scene.addRectangle([0,0,0],[-5,0,0],[0,5,0], [200,50,250])
# scene.addSphere([0,0,0], 1.0, 2, [100,200,100])

texture_array = np.asarray(Image.open("texture_example.jpg"))


number_triangles = scene.total_triangles
number_vertexs = scene.total_vertexs

vertexs = np.empty((number_vertexs,3), dtype = np.float64)
triangles = np.empty((number_triangles,3), dtype = np.uint16)
triangle_color = np.empty((number_triangles,3), dtype = np.uint8)
triangle_specular = np.empty(number_triangles, dtype = np.float32)
triangle_reflectance = np.empty(number_triangles, dtype = np.float32)


index_vertexs = 0
index_triangles = 0
index_add_triangle_vertexs = 0

for instance in scene.instances:
    
    for vertex in instance.vertexs:
        vertexs[index_vertexs] = vertex
        index_vertexs += 1

    for triangle in instance.triangles:
        
        triangles[index_triangles] = triangle + index_add_triangle_vertexs
        
        triangle_color[index_triangles] = instance.color
        triangle_specular[index_triangles] = instance.specular
        triangle_reflectance[index_triangles] = instance.reflectance
        
        index_triangles += 1
    
    index_add_triangle_vertexs += instance.number_vertexs


blank_image = 255 * np.ones((Camera.pixels_x,Camera.pixels_y,3), dtype = int) 
depth_buffer_init = 255 * np.zeros((Camera.pixels_x,Camera.pixels_y), dtype = np.float64) 

# Inicialización de GLFW
if not glfw.init():
    raise Exception("No se pudo inicializar GLFW")

# Crear una ventana con OpenGL
title = "Rasterizer"
window = glfw.create_window(pixels_x, pixels_y, title, None, None)
if not window:
    glfw.terminate()
    raise Exception("No se pudo crear la ventana")

glfw.make_context_current(window)

glfw.swap_interval(0)  # Deshabilita V-Sync - Si no es limiten els FPS a 60 Hz

# Crear una textura para cargar la imagen
def setup_texture(image):
    glEnable(GL_TEXTURE_2D)
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    
    # Configurar la textura
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    # Cargar la imagen a la textura
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.shape[1], image.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, image)

    return texture

# Dibuja la textura en la pantalla
def display_image():
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex2f(-1, -1)
    glTexCoord2f(1, 0)
    glVertex2f(1, -1)
    glTexCoord2f(1, 1)
    glVertex2f(1, 1)
    glTexCoord2f(0, 1)
    glVertex2f(-1, 1)
    glEnd()
    

# Variables para el cálculo de FPS
previous_time = glfw.get_time()
frame_count = 0
fps = 0

# Inicialitzem la textura
texture = setup_texture(blank_image)

while not glfw.window_should_close(window):
    
    # Procesar eventos - Teclat i altres events
    glfw.poll_events()
    
    # Coordenades: [Y, X, Z] = [vertical, horitzontal, profunditat]
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:  # Flecha hacia delante
        Camera.moveCamera([0, 0, 0.1])
        
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:  # Flecha hacia atras
        Camera.moveCamera([0, 0, -0.1])
        
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:  # Flecha hacia izquierda
        Camera.moveCamera([0, 0.1, 0])
        
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:  # Flecha hacia derecha
        Camera.moveCamera([0, -0.1, 0])
        
    if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:  # Flecha hacia arriba
        Camera.moveCamera([0.1, 0, 0])
        
    if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:  # Flecha hacia abajo
        Camera.moveCamera([-0.1, 0, 0])
        
    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
        Camera.intrinsicRotation(np.pi/30)
        
    if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
        Camera.intrinsicRotation(-np.pi/30)
        
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        Camera.rotateScreenY(-np.pi/300)
    
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        Camera.rotateScreenY(np.pi/300)
    
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        Camera.rotateScreenX(-np.pi/300)
        
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        Camera.rotateScreenX(np.pi/300)
        
    # Tiempo actual
    current_time = glfw.get_time()
    delta_time = current_time - previous_time
    frame_count += 1

    # Actualizar FPS cada segundo (es pot canviar)
    if delta_time >= 1:
        
        fps = frame_count / delta_time
        previous_time = current_time
        frame_count = 0
        
        glfw.set_window_title(window, title + "- FPS: " + str(fps))

    # Netegem el buffer de color
    glClear(GL_COLOR_BUFFER_BIT)
    
    # Generem la nova imatge
    inputs = (Camera.camera_pos, Camera.d, Camera.len_x, Camera.len_y, Camera.mida_pixel_x, Camera.mida_pixel_y)
    
    image = print_image(np.copy(blank_image), np.copy(depth_buffer_init), texture_array, Camera.changeBaseMatrix, vertexs, triangles, triangle_color, triangle_specular, triangle_reflectance, *inputs)
    
    # Apliquem la nova imatge a la textura
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.shape[1], image.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, image)

    # Actualitzem la textura
    display_image()

    # Actualitzem la finestra - Carrega la textura que esta en el buffer de la pantalla
    glfw.swap_buffers(window)

# Terminar la aplicación
glfw.destroy_window(window)
glfw.terminate()

