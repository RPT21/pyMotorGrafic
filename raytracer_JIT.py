from cameraObj import camera
import numpy as np
from numba import njit, jit
import numba
import math
from Scene_Obj import Scene, Triangle, Rectangle, RectanglePrisma
import glfw
from OpenGL.GL import *
from OpenGL.GLUT import *


# ////////////////////////////////////// INIT /////////////////////////////////////////////

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

scene.addInstance(RectanglePrisma([-1,1,0], [1,1,0], [-1,-1,0], [-1,1,5], [0,0,255], -1, 0.5))
scene.addInstance(Rectangle([-10,-10,0],[10,-10,0],[-10,10,0], [200,50,250], -1, 0))

# scene.addTriangle([0,0,0],[0,5,0],[0,0,6], [42,233,250], 500)
# scene.addTriangle([-0.5,0,0],[1,5,0],[1,0,6], [200,50,250])
# scene.addSphere([0,0,0], 1.0, 1, [100,200,100], 500)
# scene.addSphere([0,0,0], 1.0, 1, [100,200,100], 500)

scene.addLight([5,5,5], "puntual", 0.6)
scene.addLight([-5,5,5], "directional", 0.2)
scene.addLight([0,0,0], "ambiental", 0.2)

# Definim l'imatge que pintarem:
image = 255 * np.ones((Camera.pixels_x,Camera.pixels_y,3), dtype = np.uint8) 

# ////////////////////////////////////// CALCULATE /////////////////////////////////////////////

number_triangles = scene.total_triangles
number_vertexs = scene.total_vertexs

vertexs = np.empty((number_vertexs,3), dtype = np.float64)
triangles = np.empty((number_triangles,3), dtype = np.uint16)
triangle_color_matrix = np.empty((number_triangles,3), dtype = np.float64) # Per fer una multiplicacio de un vector de color amb un escalar, ha de ser de tipus float
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
        
        triangle_color_matrix[index_triangles] = instance.color
        triangle_specular[index_triangles] = instance.specular
        triangle_reflectance[index_triangles] = instance.reflectance
        
        index_triangles += 1
    
    index_add_triangle_vertexs += instance.number_vertexs
    
    
triangle_matrix = np.empty((3,3, scene.total_triangles))
n = 0
for triangle in triangles:
    for v in range(3):
        triangle_matrix[v,:,n] = vertexs[triangle[v]]
    n += 1
    
    
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


@jit
def init_triangle_intersection(triangle_matrix):
    
    number_triangles = triangle_matrix.shape[2]
    
    changeBase_matrix = np.empty((3,3,number_triangles))
    triangle_plane_matrix = np.empty((number_triangles, 4))

    for n in range(number_triangles):
        _triangle_ = triangle_matrix[:,:,n]
        
        At, Bt, Ct = _triangle_[0], _triangle_[1], _triangle_[2]
        
        # Vectors del triangle
        ABt = Bt - At
        ACt = Ct - At
        
        vect_direct_tri = np.cross(ACt, ABt) # Important l'ordre del producte per calcular l'angle despres!
        vect_direct_tri = vect_direct_tri / np.linalg.norm(vect_direct_tri)
        
        # Pla del triangle:
        A = vect_direct_tri[0]
        B = vect_direct_tri[1]
        C = vect_direct_tri[2]
        D = -(A*At[0] + B*At[1] + C*At[2])
        
        # Carreguem a la matriu l'equacio del pla
        triangle_plane_matrix[n,0] = A
        triangle_plane_matrix[n,1] = B
        triangle_plane_matrix[n,2] = C
        triangle_plane_matrix[n,3] = D
        
        # Matriu canvi de base:
        M = np.empty((3,3))
        M[:,0] = ABt
        M[:,1] = ACt
        M[:,2] = vect_direct_tri
        M_inv = np.linalg.inv(M)
        
        # Carreguem la matriu canvi de base:
        changeBase_matrix[:,:,n] = M_inv

        
    return changeBase_matrix, triangle_plane_matrix
        

changeBase_matrix, triangle_plane_matrix = init_triangle_intersection(triangle_matrix)

img_x = Camera.pixels_x
img_y = Camera.pixels_y
len_x = Camera.len_x
len_y = Camera.len_y
pixel_len_x = Camera.mida_pixel_x
pixel_len_y = Camera.mida_pixel_x

number_pixels = img_x*img_y
number_triangles = triangle_matrix.shape[2]
background_color = np.array([0,0,0], dtype=np.uint8)

@jit
def TraceRay(Point, RayDirection, t_min, t_max, triangle_matrix, triangle_plane_matrix, changeBase_matrix, triangle_color_matrix, triangle_specular, triangle_reflectance, recursion_depth):
    
    Intersection = TriangleIntersection(Point, RayDirection, t_min, t_max, triangle_matrix, triangle_plane_matrix, changeBase_matrix)
    
    inter_array = Intersection[0:3]
    nearest_triangle = int(Intersection[3])
    
    if nearest_triangle == -1:
        return background_color
    
    else:
        
        # Important, no podem passar un vector a la vegada. S'ha de passar component a component:
        A = triangle_plane_matrix[nearest_triangle, 0]
        B = triangle_plane_matrix[nearest_triangle, 1]
        C = triangle_plane_matrix[nearest_triangle, 2]
        
        Vx = RayDirection[0]
        Vy = RayDirection[1]
        Vz = RayDirection[2]
        
        pixel_intensity = 0
        
        # Anem a veure si estem mirant el triangle per davant o per darrere:
        # Calculem el producte escalar, amb el vector que va del punt de intersecció fins a la camara
        # i el vector director del triangle.
        
        V_mod = math.sqrt(Vx**2 + Vy**2 + Vz**2)
        Ux_C = -Vx / V_mod
        Uy_C = -Vy / V_mod
        Uz_C = -Vz / V_mod
        
        dot_prod = A*Ux_C + B*Uy_C + C*Uz_C
        
        if dot_prod >= 0:
            isCameraFrontside = True
        else:
            isCameraFrontside = False
            
        
        for n in range(light_matrix.shape[0]):
            if light_matrix[n, 3] == 0 or light_matrix[n, 3] == 1: # LLum puntual o direccional
            
                # Ara he de saber si la llum l'enfoca per davant o per darrere
                
                if light_matrix[n, 3] == 0: # Llum puntual
                    light_posX = light_matrix[n, 0]
                    light_posY = light_matrix[n, 1]
                    light_posZ = light_matrix[n, 2]
                    
                    Lx = light_posX - inter_array[0]
                    Ly = light_posY - inter_array[1]
                    Lz = light_posZ - inter_array[2]
                    
                else: # Llum direccional
                    # La llum la definim sempre des del triangle
                    Lx = -light_matrix[n, 0]
                    Ly = -light_matrix[n, 1]
                    Lz = -light_matrix[n, 2]
                
                L_mod = math.sqrt(Lx**2 + Ly**2 + Lz**2)
                
                Ux_L = Lx / L_mod
                Uy_L = Ly / L_mod
                Uz_L = Lz / L_mod
                
                dot_prod_light = A*Ux_L + B*Uy_L + C*Uz_L
                

                if dot_prod_light >= 0:
                    isLightFrontside = True
                else:
                    isLightFrontside = False
                    
                # Comprovem si estem mirant la cara iluminada per la llum
                if isCameraFrontside == isLightFrontside:
                    
                    # Shadows - Hem de mirar si hi ha algun objecte entre el punt de interseccio i la llum
                    # Si hi ha alguna interseccio, aquell punt no esta rebent llum de la font de llum considerada.
                    
                    # Fem un petit increment per tenir el triangle des del qual tracem el raig darrera i no el considerem.
                    if light_matrix[n, 3] == 0: # Llum puntual - Si el triangle intersectat esta entre el punt i la llum puntual:
                        Intersection_L = TriangleIntersection(inter_array, [Lx,Ly,Lz], 0.0001, 1, triangle_matrix, triangle_plane_matrix, changeBase_matrix)
                    else: # Si la llum es direccional hem de considerar qualsevol triangle, ja que la llum ilumina des del infinit
                        Intersection_L = TriangleIntersection(inter_array, [Lx,Ly,Lz], 0.0001, np.inf, triangle_matrix, triangle_plane_matrix, changeBase_matrix)
                    
                    nearest_triangle_L = int(Intersection_L[3])
                    
                    if nearest_triangle_L == -1: # Esta iluminat
                        # Llum difusiva:
                        light_intesity = light_matrix[n, 4]
                        pixel_intensity += light_intesity * abs(dot_prod_light)
                        
                        s = triangle_specular[nearest_triangle]
                        r = triangle_reflectance[nearest_triangle]
                        
                        # Llum especular:
                        if s != -1 or r > 0:
                            
                            Rx = 2*A*dot_prod_light - Ux_L
                            Ry = 2*B*dot_prod_light - Uy_L
                            Rz = 2*C*dot_prod_light - Uz_L
                            
                            R_mod = math.sqrt(Rx**2 + Ry**2 + Rz**2)
                            
                            Ux_R = Rx / R_mod
                            Uy_R = Ry / R_mod
                            Uz_R = Rz / R_mod
                            
                            if s != -1:
                                dot_prod_Reflect_Camera = Ux_R*Ux_C + Uy_R*Uy_C + Uz_R*Uz_C
                                pixel_intensity += light_intesity * math.pow(abs(dot_prod_Reflect_Camera), s)

            elif light_matrix[n, 3] == 2: # LLum ambiental
                
                pixel_intensity += light_matrix[n, 4]
                
    if pixel_intensity > 1:
        pixel_intensity = 1
        
    local_color = np.round(triangle_color_matrix[nearest_triangle] * pixel_intensity).astype(np.uint8)
    
    
    if recursion_depth <= 0 or r <= 0:
        return local_color
    
    reflected_color = TraceRay(inter_array, [Rx, Ry, Rz], 0.0001, np.inf, triangle_matrix, triangle_plane_matrix, changeBase_matrix, triangle_color_matrix, triangle_specular, triangle_reflectance, recursion_depth - 1)
    
    return np.round(local_color * (1 - r) + reflected_color * r).astype(np.uint8)
        
        

@jit
def TriangleIntersection(Point, RayDirection, t_min, t_max, triangle_matrix, triangle_plane_matrix, changeBase_matrix):
    
    t_intersect = np.inf
    return_array = np.zeros(4, dtype=np.float64)
    return_array[3] = -1

    for m in range(number_triangles):
        
        # Vertex A del triangle
        At = triangle_matrix[0,:,m]
        
        # Pla del triangle
        A, B, C, D = triangle_plane_matrix[m, 0], triangle_plane_matrix[m, 1], triangle_plane_matrix[m, 2], triangle_plane_matrix[m, 3]
        
        # Paramétricas de la recta: x = x0 + t * dx, y = y0 + t * dy, z = z0 + t * dz
        # Sustitución en la ecuación del plano: A*(x0 + t*dx) + B*(y0 + t*dy) + C*(z0 + t*dz) + D = 0
        # Resolvemos para t (o sigui, aillem t de la equacio anterior):
        numerador = -(A * Point[0] + B * Point[1] + C * Point[2] + D)
        denominador = A * RayDirection[0] + B * RayDirection[1] + C * RayDirection[2]

        # Calculamos t
        if denominador == 0:
            t = np.nan
        else:
            t = numerador / denominador
        
        # Calculem els components dels punts de intersecció
        inter_x = Point[0] + t * RayDirection[0]
        inter_y = Point[1] + t * RayDirection[1]
        inter_z = Point[2] + t * RayDirection[2]
        
        # Fem el canvi de base per veure'l en aquesta base:
        M_inv = changeBase_matrix[:,:,m]
        
        # Canvi a la base centrada en el vertex A del triangle i que te com a vectors els costats del triangle
        alfa = (inter_x - At[0]) * M_inv[0, 0] + (inter_y - At[1]) * M_inv[0, 1] + (inter_z - At[2]) * M_inv[0, 2]
        beta = (inter_x - At[0]) * M_inv[1, 0] + (inter_y - At[1]) * M_inv[1, 1] + (inter_z - At[2]) * M_inv[1, 2]
                    
        # Comprovem si hem tocat el triangle:
        if alfa >= 0 and beta >= 0 and alfa + beta <= 1:
            if t > t_min and t < t_max:
                if t < t_intersect:
                    
                    # Guardem el triangle més pròxim
                    t_intersect = t
                    
                    # Guardem el triangle i la interseccio per mes endevant
                    return_array[0] = inter_x
                    return_array[1] = inter_y
                    return_array[2] = inter_z
                    return_array[3] = m # Nearest triangle
                    
                    
    return return_array

@jit
def print_image(image, light_matrix, triangle_matrix, changeBase_matrix, triangle_plane_matrix, triangle_color_matrix, triangle_specular, triangle_reflectance,
                screen_point, base_x, base_y, camera_pos):
    
    # Punt de la recta, es la posicio de la camera sempre:
    Cx, Cy, Cz = camera_pos[0], camera_pos[1], camera_pos[2]

    # Inici de la pantalla, ens situem en la punta oposada de la pantalla al sistema de coordenades.
    pantalla_init_x = screen_point[0] - (len_x / 2 * base_x[0]) - (len_y / 2 * base_y[0])
    pantalla_init_y = screen_point[1] - (len_x / 2 * base_x[1]) - (len_y / 2 * base_y[1])
    pantalla_init_z = screen_point[2] - (len_x / 2 * base_x[2]) - (len_y / 2 * base_y[2])
    
    
    for Px in range(image.shape[0]):
        for Py in range(image.shape[1]):
                    
            # Calculem la posicio del pixel
            pixelPos_x = pantalla_init_x + Px * pixel_len_x * base_x[0] + Py * pixel_len_y * base_y[0]
            pixelPos_y = pantalla_init_y + Px * pixel_len_x * base_x[1] + Py * pixel_len_y * base_y[1]
            pixelPos_z = pantalla_init_z + Px * pixel_len_x * base_x[2] + Py * pixel_len_y * base_y[2]
            
            # Calculem el vector director de la recta que passa per l'origen i el pixel
            Vx = pixelPos_x - Cx
            Vy = pixelPos_y - Cy
            Vz = pixelPos_z - Cz

            image[Py, Px] = TraceRay(camera_pos, [Vx,Vy,Vz], 0, np.inf, triangle_matrix, triangle_plane_matrix, changeBase_matrix, triangle_color_matrix, triangle_specular, triangle_reflectance, 3)



# ////////////////////////////////////// PRINT /////////////////////////////////////////////

_input_ = (image,
           light_matrix,
           triangle_matrix,
           changeBase_matrix,
           triangle_plane_matrix,
           triangle_color_matrix,
           triangle_specular,
           triangle_reflectance)


# Inicialización de GLFW
if not glfw.init():
    raise Exception("No se pudo inicializar GLFW")

# Crear una ventana con OpenGL
title = "Raytracer"
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
texture = setup_texture(image)

while not glfw.window_should_close(window):
    
    # Procesar eventos - Teclat i altres events
    glfw.poll_events()

    # Coordenades: [Y, X, Z] = [vertical, horitzontal, profunditat]
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:  # Flecha hacia delante
        Camera.moveCamera([0, 0, 0.5])
        
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:  # Flecha hacia atras
        Camera.moveCamera([0, 0, -0.5])
        
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:  # Flecha hacia izquierda
        Camera.moveCamera([0, 0.5, 0])
        
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:  # Flecha hacia derecha
        Camera.moveCamera([0, -0.5, 0])
        
    if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:  # Flecha hacia arriba
        Camera.moveCamera([0.5, 0, 0])
        
    if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:  # Flecha hacia abajo
        Camera.moveCamera([-0.5, 0, 0])
        
    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
        Camera.intrinsicRotation(np.pi/10)
        
    if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
        Camera.intrinsicRotation(-np.pi/10)
        
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        Camera.rotateScreenY(-np.pi/25)
    
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        Camera.rotateScreenY(np.pi/25)
    
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        Camera.rotateScreenX(-np.pi/25)
        
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        Camera.rotateScreenX(np.pi/25)
        
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
    
    # Actualitzem les dades de la camera
    screen_point = Camera.punt_pla
    base_x = Camera.coordX_pantalla
    base_y = Camera.coordY_pantalla
    camera_pos = Camera.camera_pos
    
    # Generem la nova imatge
    print_image(*_input_, screen_point, base_x, base_y, camera_pos)
    
    # Apliquem la nova imatge a la textura
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.shape[1], image.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, image)

    # Actualitzem la textura
    display_image()

    # Actualitzem la finestra - Carrega la textura que esta en el buffer de la pantalla
    glfw.swap_buffers(window)

# Terminar la aplicación
glfw.destroy_window(window)
glfw.terminate()