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

# ////////////////////////////// INIT /////////////////////////////////////////    
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

# scene.addLight([5,5,5], "puntual", 0.6)
scene.addLight([0.5,1,1], "puntual", 0.8)
scene.addLight([-5,5,5], "directional", 0.1)
scene.addLight([0,0,0], "ambiental", 0.1)

texture_array = np.asarray(Image.open("texture_example.jpg"))

number_triangles = scene.total_triangles
number_vertexs = scene.total_vertexs

vertexs = np.empty((number_vertexs,3), dtype = np.float64)
triangles = np.empty((number_triangles,3), dtype = np.uint16)
triangle_color = np.empty((number_triangles,3), dtype = np.uint8)
triangle_specular = np.empty(number_triangles, dtype = np.float32)
triangle_reflectance = np.empty(number_triangles, dtype = np.float32)
triangle_texels = np.empty((number_triangles,3,2))
triangle_vertexs_normal = np.empty((number_triangles, 3, 3))


T0X = 0.2
T0Y = 0.2

T1X = 1
T1Y = 0.2

T2X = 0.5
T2Y = 1

triangle_texels[0,0,0] = T0X
triangle_texels[0,0,1] = T0Y

triangle_texels[0,1,0] = T1X
triangle_texels[0,1,1] = T1Y

triangle_texels[0,2,0] = T2X
triangle_texels[0,2,1] = T2Y


index_vertexs = 0
index_triangles = 0
index_add_triangle_vertexs = 0
index_vertex_normal = 0

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
        
    for vertex_normal in instance.triangle_vertexs_normal:
        triangle_vertexs_normal[index_vertex_normal] = vertex_normal
        index_vertex_normal += 1
        
    index_add_triangle_vertexs += instance.number_vertexs
    
light_matrix = np.empty((len(scene.lights),5))
    
n = 0
for light in scene.lights: 
    light_matrix[n, 0:3] = light["parameter"]
    if light["type"] == "puntual":
        light_matrix[n, 3] = 0
    elif light["type"] == "directional":
        light_matrix[n, 3] = 1
    elif light["type"] == "ambiental":
        light_matrix[n, 3] = 2
        
    light_matrix[n, 4] = light["intensity"]

    n += 1
    
# /////////////////////////// FUNCTIONS ///////////////////////////////////////

def print_image(image, depth_buffer, texture_array, changeBaseMatrix, 
                clippingPlanes, vertexs, triangle_texels, triangles, 
                triangle_color, triangle_specular, triangle_reflectance, 
                triangle_vertexs_normal, light_matrix, 
                camera_pos, d, len_x, len_y, len_pixelX, len_pixelY):
 
    changeBaseMatrix = np.ascontiguousarray(changeBaseMatrix)
    vertexs = np.ascontiguousarray(vertexs)
    
    changeBase_vertexs = np.copy(vertexs)
    changeBase_triangle_vertexs_normal = np.empty_like(triangle_vertexs_normal)
    changeBase_light_matrix = np.empty_like(light_matrix)
    
    # ///////////////////////////// CHANGE BASE ///////////////////////////////
    
    # Efectuem la translacio a la base de la camara
    changeBase_vertexs = changeBase_vertexs - camera_pos
    
    # Fem el canvi de base:
    for n in range(vertexs.shape[0]):
        changeBase_vertexs[n] = np.dot(changeBaseMatrix, changeBase_vertexs[n])
        
    # Hem de fer el mateix per les normals dels triangles:
    for n in range(triangle_vertexs_normal.shape[0]):
        changeBase_triangle_vertexs_normal[n,0] = np.dot(changeBaseMatrix, triangle_vertexs_normal[n,0])
        changeBase_triangle_vertexs_normal[n,1] = np.dot(changeBaseMatrix, triangle_vertexs_normal[n,1])
        changeBase_triangle_vertexs_normal[n,2] = np.dot(changeBaseMatrix, triangle_vertexs_normal[n,2])
        
    # Per la llum tambe:
    for n in range(light_matrix.shape[0]):
        changeBase_light_matrix[n] = light_matrix[n]
        
        if changeBase_light_matrix[n,3] == 0:
            changeBase_light_matrix[n,0:3] = changeBase_light_matrix[n,0:3] - camera_pos
            
        changeBase_light_matrix[n,0:3] = np.dot(changeBaseMatrix, changeBase_light_matrix[n,0:3])
    
    # ////////////////////////////// CLIPPING /////////////////////////////////
    
    new_vertexs = np.empty((0,3))
    new_triangle_color = np.empty((0,3))
    new_triangles = np.empty((0,3))
    new_triangle_texels = np.empty((0,3,2))
    new_triangle_vertexs_normal = np.empty((0,3,3))
    
    # Calculem els nous vertexs:
    for n in range(triangles.shape[0]):
        
        indexA = triangles[n,0]
        indexB = triangles[n,1]
        indexC = triangles[n,2]
        color = triangle_color[n]
        
        triangle_vertexs = np.array([[changeBase_vertexs[indexA], changeBase_vertexs[indexB], changeBase_vertexs[indexC]]])
        
        new_vertexs, new_triangles, new_triangle_color, new_triangle_texels, new_triangle_vertexs_normal = \
        ClipTriangle(triangle_vertexs, color, clippingPlanes, new_vertexs, new_triangles, new_triangle_color, triangle_texels, changeBase_triangle_vertexs_normal)
        
    if new_vertexs.shape[0] == 0:
        return image
    
    # ////////////////////////////// PROJECTION ///////////////////////////////
    
    # w = z /d on d es la distancia respecte el pla de projeccio, i z es la coordenada del punt que volem projectar.
    # Suposem que estem al origen, i el pla de projeccio esta a z = d
    # Les coordenades en projeccio perspectiva son: x' = x / w ; y' = y / w; z' = d
    # Les coordenades fan que el pla de projeccio estigui al pla XY
    
    # El z_axis es super important, ja que al ser una coordenada perpendicular al pla de projeccio, ens diu la distancia de cada punt respecte el pla
    # Aixo ens permetra calcular correctament el Z-buffering. Un error que he fet era calcular el modul de distancia respecte el punt de projeccio
    # Aquest calcul no em servia per saber la profunditat de cada punt, ja que el que importa es la distancia respecte el pla de projeccio
    z_axis = new_vertexs[:,2]
    
    # Cada vertex te la seva w - parametre de projeccio, ens permet calcular la projeccio dels punts al pla de projeccio
    w = z_axis / d
    
    projected_vertexs = new_vertexs / w[:, np.newaxis]

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
    
    for n in range(new_triangles.shape[0]):
        
        indexA = new_triangles[n,0]
        indexB = new_triangles[n,1]
        indexC = new_triangles[n,2]
        
        color = new_triangle_color[n]
        vertex_normals = new_triangle_vertexs_normal[n]
            
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
                      new_triangle_texels[n,0,0],new_triangle_texels[n,0,1],
                      new_triangle_texels[n,1,0],new_triangle_texels[n,1,1],
                      new_triangle_texels[n,2,0],new_triangle_texels[n,2,1],
                      changeBase_light_matrix,
                      projected_vertexs[indexA],
                      projected_vertexs[indexB],
                      projected_vertexs[indexC],
                      vertex_normals,
                      d,
                      phongShading=True,
                      useTexture=True)
            
    return image

def SignedDistance(point, plane):
    return point[0]*plane[0] + point[1]*plane[1] + point[2]*plane[2] + plane[3]

def ClipTriangle(clipped_triangles, color, planes, new_vertexs, new_triangles, new_triangle_color, triangle_texels, triangle_vertexs_normal):
    # Aquesta funcio pot retornar 0, 1 o 2 triangles
    for plane in planes:
        clipped_triangles, triangle_texels, triangle_vertexs_normal = ClipTrianglesAgainstPlane(clipped_triangles, triangle_texels, triangle_vertexs_normal, plane)
        if clipped_triangles.shape[0] == 0:
            return new_vertexs, new_triangles, new_triangle_color, triangle_texels, triangle_vertexs_normal
        
    index_vertexs = new_vertexs.shape[0]
    
    for triangle in clipped_triangles:
        
        new_vertexs = np.vstack((new_vertexs, triangle[0]))
        new_vertexs = np.vstack((new_vertexs, triangle[1]))
        new_vertexs = np.vstack((new_vertexs, triangle[2]))
        
        new_triangles = np.vstack((new_triangles, [index_vertexs, index_vertexs + 1, index_vertexs + 2])).astype(int)
        
        new_triangle_color = np.vstack((new_triangle_color, color))
        new_triangle_color = np.vstack((new_triangle_color, color))
        new_triangle_color = np.vstack((new_triangle_color, color))
        
        index_vertexs += 3
        
    return new_vertexs, new_triangles, new_triangle_color, triangle_texels, triangle_vertexs_normal

        
def ClipTrianglesAgainstPlane(clipped_triangles, triangle_texels, triangle_vertexs_normal, plane):
    
    returned_triangles = np.empty((0,3,3))
    returned_texels = np.empty((0,3,2))
    returned_triangle_vertexs_normal = np.empty((0,3,3))
    
    k = 0
    for triangle in clipped_triangles:
        A = triangle[0]
        B = triangle[1]
        C = triangle[2]
        
        # Test per comprovar que estem analitzant un triangle i no un segment
        area_indicator = (B[1] - C[1])*(A[0] - C[0]) + (C[0] - B[0])*(A[1] - C[1])
        if area_indicator == 0:
            return returned_triangles, returned_texels, returned_triangle_vertexs_normal
        
        T0 = triangle_texels[k, 0]
        T1 = triangle_texels[k, 1]
        T2 = triangle_texels[k, 2]
        
        vertex_normals = triangle_vertexs_normal[k]
        
        k += 1
        
        d0 = SignedDistance(A, plane)
        d1 = SignedDistance(B, plane)
        d2 = SignedDistance(C, plane)
        
        if d0 < 0 and d1 < 0 and d2 < 0:
            continue
        
        if d0 > 0 and d1 > 0 and d2 > 0:
            returned_triangles = np.vstack((returned_triangles, [[A,B,C]]))
            returned_texels = np.vstack((returned_texels, [[T0,T1,T2]]))
            returned_triangle_vertexs_normal = np.vstack((returned_triangle_vertexs_normal, [vertex_normals]))
            continue
        
        # d0 >= d1 >= d2
        if d1 > d0:
            d0, d1 = d1, d0
            A, B = B, A
            T0, T1 = T1, T0
        if d2 > d0:
            d0, d2 = d2, d0
            A, C = C, A
            T0, T2 = T2, T0
        if d2 > d1:
            d1, d2 = d2, d1
            B, C = C, B
            T1, T2 = T2, T1
        
        # Si tenim 1 punt dintre i dos punts fora:
        if d0 > 0 and d1 < 0:
            # Calculem els punts de interseccio amb el pla
            AB = B - A
            AC = C - A
            
            numerador = -(plane[0] * A[0] + plane[1] * A[1] + plane[2] * A[2] + plane[3])
            denominador = plane[0] * AB[0] + plane[1] * AB[1] + plane[2] * AB[2]
            
            t = numerador / denominador
            
            B_prima = A + t*AB
            T1_prima = t*T1 + (1-t)*T0
            
            numerador = -(plane[0] * A[0] + plane[1] * A[1] + plane[2] * A[2] + plane[3])
            denominador = plane[0] * AC[0] + plane[1] * AC[1] + plane[2] * AC[2]
            
            t = numerador / denominador
            
            C_prima = A + t*AC
            T2_prima = t*T2 + (1-t)*T0
            
            returned_triangles = np.vstack((returned_triangles, [[A,B_prima,C_prima]]))
            returned_texels = np.vstack((returned_texels, [[T0,T1_prima,T2_prima]]))
            returned_triangle_vertexs_normal = np.vstack((returned_triangle_vertexs_normal, [vertex_normals]))
    
        # Si tenim dos punts dintre i 1 punt fora, C es el que esta a fora:
        else:
            
            # Calculem els punts de interseccio amb el pla
            AC = C - A
            BC = C - B
            
            numerador = -(plane[0] * A[0] + plane[1] * A[1] + plane[2] * A[2] + plane[3])
            denominador = plane[0] * AC[0] + plane[1] * AC[1] + plane[2] * AC[2]
            
            t = numerador / denominador
            
            A_prima = A + t*AC
            T0_prima = t*T2 + (1-t)*T0
            
            numerador = -(plane[0] * B[0] + plane[1] * B[1] + plane[2] * B[2] + plane[3])
            denominador = plane[0] * BC[0] + plane[1] * BC[1] + plane[2] * BC[2]
            
            t = numerador / denominador
            
            B_prima = B + t*BC
            T1_prima = t*T2 + (1-t)*T1
            
            returned_triangles = np.vstack((returned_triangles, [[A, A_prima, B]]))
            returned_triangles = np.vstack((returned_triangles, [[A_prima, B_prima, B]]))
            
            returned_texels = np.vstack((returned_texels, [[T0,T0_prima,T1]]))
            returned_texels = np.vstack((returned_texels, [[T0_prima,T1_prima,T1]]))
            
            returned_triangle_vertexs_normal = np.vstack((returned_triangle_vertexs_normal, [vertex_normals]))
            returned_triangle_vertexs_normal = np.vstack((returned_triangle_vertexs_normal, [vertex_normals]))
            
    return returned_triangles, returned_texels, returned_triangle_vertexs_normal


# /////////////////////////////// PRINT ///////////////////////////////////////

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
    
    image = print_image(np.copy(blank_image), 
                        np.copy(depth_buffer_init),
                        texture_array, 
                        Camera.changeBaseMatrix, 
                        Camera.clippingPlanes, 
                        vertexs, 
                        triangle_texels, 
                        triangles, 
                        triangle_color, 
                        triangle_specular, 
                        triangle_reflectance, 
                        triangle_vertexs_normal, 
                        light_matrix,
                        *inputs)
    
    # Apliquem la nova imatge a la textura
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.shape[1], image.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, image)

    # Actualitzem la textura
    display_image()

    # Actualitzem la finestra - Carrega la textura que esta en el buffer de la pantalla
    glfw.swap_buffers(window)

# Terminar la aplicación
glfw.destroy_window(window)
glfw.terminate()

