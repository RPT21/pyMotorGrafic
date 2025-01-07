from cameraObj import camera
import numpy as np
from numba import njit
from numba import cuda, guvectorize
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

# scene.addInstance(RectanglePrisma([-1,1,0], [1,1,0], [-1,-1,0], [-1,1,5], [0,0,255], 10, 0))
# scene.addInstance(Rectangle([-10,-10,0],[10,-10,0],[-10,10,0], [200,50,250], -1, 0))
scene.addInstance(Triangle([0,0,0],[0,5,0],[0,0,6], [42,233,250], -1, 0))
scene.addInstance(Triangle([-0.5,0,0],[1,5,0],[1,0,6], [200,50,250], -1, 0))



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
triangle_color_matrix = np.empty((number_triangles,3), dtype = np.uint8)
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


@njit
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
background_color = np.array([255,255,255])

@cuda.jit
def print_image(image, light_matrix, triangle_matrix, changeBase_matrix, triangle_plane_matrix, triangle_color_matrix, triangle_specular, triangle_reflectance,
                screen_point, base_x, base_y, camera_pos):
    
    Px, Py = cuda.grid(2)
    
    # Verificar limits de la imatge
    if Px >= image.shape[0] or Py >= image.shape[1]:
        return
    
    # Array per guardar les dades de la interseccio amb el triangle mes proxim
    inter_array = cuda.local.array(shape=3,dtype=numba.float64)

    # Punt de la recta, es la posicio de la camera sempre:
    Cx, Cy, Cz = camera_pos[0], camera_pos[1], camera_pos[2]

    # Inici de la pantalla, ens situem en la punta oposada de la pantalla al sistema de coordenades.
    pantalla_init_x = screen_point[0] - (len_x / 2 * base_x[0]) - (len_y / 2 * base_y[0])
    pantalla_init_y = screen_point[1] - (len_x / 2 * base_x[1]) - (len_y / 2 * base_y[1])
    pantalla_init_z = screen_point[2] - (len_x / 2 * base_x[2]) - (len_y / 2 * base_y[2])
            
    # Calculem la posicio del pixel
    pixelPos_x = pantalla_init_x + Px * pixel_len_x * base_x[0] + Py * pixel_len_y * base_y[0]
    pixelPos_y = pantalla_init_y + Px * pixel_len_x * base_x[1] + Py * pixel_len_y * base_y[1]
    pixelPos_z = pantalla_init_z + Px * pixel_len_x * base_x[2] + Py * pixel_len_y * base_y[2]
    
    # Calculem el vector director de la recta que passa per l'origen i el pixel
    Vx = pixelPos_x - Cx
    Vy = pixelPos_y - Cy
    Vz = pixelPos_z - Cz
    
    t_min = np.inf
    nearest_triangle = -1

    for m in range(number_triangles):
        
        # Vertex A del triangle
        At = triangle_matrix[0,:,m]
        
        # Pla del triangle
        A, B, C, D = triangle_plane_matrix[m, 0], triangle_plane_matrix[m, 1], triangle_plane_matrix[m, 2], triangle_plane_matrix[m, 3]
        
        # Paramétricas de la recta: x = x0 + t * dx, y = y0 + t * dy, z = z0 + t * dz
        # Sustitución en la ecuación del plano: A*(x0 + t*dx) + B*(y0 + t*dy) + C*(z0 + t*dz) + D = 0
        # Resolvemos para t (o sigui, aillem t de la equacio anterior):
        numerador = -(A * Cx + B * Cy + C * Cz + D)
        denominador = A * Vx + B * Vy + C * Vz

        # Calculamos t
        if denominador == 0:
            t = np.nan
        else:
            t = numerador / denominador
        
        # Calculem els components dels punts de intersecció
        inter_x = Cx + t * Vx
        inter_y = Cy + t * Vy
        inter_z = Cz + t * Vz
        
        # Fem el canvi de base per veure'l en aquesta base:
        M_inv = changeBase_matrix[:,:,m]
        
        # Canvi a la base centrada en el vertex A del triangle i que te com a vectors els costats del triangle
        alfa = (inter_x - At[0]) * M_inv[0, 0] + (inter_y - At[1]) * M_inv[0, 1] + (inter_z - At[2]) * M_inv[0, 2]
        beta = (inter_x - At[0]) * M_inv[1, 0] + (inter_y - At[1]) * M_inv[1, 1] + (inter_z - At[2]) * M_inv[1, 2]
                    
        # Comprovem si hem tocat el triangle:
        if alfa >= 0 and beta >= 0 and alfa + beta <= 1 and t > 0:
        
            # Guardem el triangle més pròxim
            if t < t_min:

                t_min = t
                
                # Guardem el triangle i la interseccio per mes endevant
                nearest_triangle = m
                inter_array[0] = inter_x
                inter_array[1] = inter_y
                inter_array[2] = inter_z
                
            
    if nearest_triangle != -1:
        
        # Important, no podem passar un vector a la vegada. S'ha de passar component a component:
            
        A = triangle_plane_matrix[nearest_triangle, 0]
        B = triangle_plane_matrix[nearest_triangle, 1]
        C = triangle_plane_matrix[nearest_triangle, 2]
        D = triangle_plane_matrix[nearest_triangle, 3]
        
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
            
            
        pixel_intensity = 0
        
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
                    epsilon = 0.0001
                    
                    inter_xL = inter_array[0] + Ux_L*epsilon
                    inter_yL = inter_array[1] + Uy_L*epsilon
                    inter_zL = inter_array[2] + Uz_L*epsilon
                    
                    for k in range(number_triangles):
                            
                        # Vertex A del triangle
                        At = triangle_matrix[0,:,k]
                        
                        # Pla del triangle
                        Al, Bl, Cl, Dl = triangle_plane_matrix[k, 0], triangle_plane_matrix[k, 1], triangle_plane_matrix[k, 2], triangle_plane_matrix[k, 3]
                        
                        
                        # Paramétricas de la recta: x = x0 + t * dx, y = y0 + t * dy, z = z0 + t * dz
                        # Sustitución en la ecuación del plano: A*(x0 + t*dx) + B*(y0 + t*dy) + C*(z0 + t*dz) + D = 0
                        # Resolvemos para t (o sigui, aillem t de la equacio anterior):
                        numerador = -(Al * inter_xL + Bl * inter_yL + Cl * inter_zL + Dl)
                        denominador = Al * Lx + Bl * Ly + Cl * Lz

                        # Calculamos t
                        if denominador == 0:
                            t = np.nan
                        else:
                            t = numerador / denominador
                        
                        # Calculem els components dels punts de intersecció
                        inter_x = inter_xL + t * Lx
                        inter_y = inter_yL + t * Ly
                        inter_z = inter_zL + t * Lz
                        
                        # Fem el canvi de base per veure'l en aquesta base:
                        M_inv = changeBase_matrix[:,:,k]
                        
                        # Canvi a la base centrada en el vertex A del triangle i que te com a vectors els costats del triangle
                        alfa = (inter_x - At[0]) * M_inv[0, 0] + (inter_y - At[1]) * M_inv[0, 1] + (inter_z - At[2]) * M_inv[0, 2]
                        beta = (inter_x - At[0]) * M_inv[1, 0] + (inter_y - At[1]) * M_inv[1, 1] + (inter_z - At[2]) * M_inv[1, 2]
                        
                        # Comprovem si hem tocat el triangle:
                        if alfa >= 0 and beta >= 0 and alfa + beta <= 1 and t > 0:
                        
                            if light_matrix[n, 3] == 0: # Llum puntual
                                if t <= 1:
                                    # Si el triangle intersectat esta entre el punt i la llum puntual:
                                    isInshadow = True
                                    break
                            else:
                                # Si la llum es direccional hem de considerar qualsevol triangle, ja que la llum ilumina des del infinit
                                isInshadow = True
                                break
                        else:
                            isInshadow = False
                            
                        
                    if isInshadow == False:
                        # Llum difusiva:
                        light_intesity = light_matrix[n, 4]
                        pixel_intensity += light_intesity * abs(dot_prod_light)
                        
                        # Llum especular:
                        s = triangle_specular[nearest_triangle]
    
                        if s != -1:
                            Rx = 2*A*dot_prod_light - Ux_L
                            Ry = 2*B*dot_prod_light - Uy_L
                            Rz = 2*C*dot_prod_light - Uz_L
                            
                            R_mod = math.sqrt(Rx**2 + Ry**2 + Rz**2)
                            
                            Ux_R = Rx / R_mod
                            Uy_R = Ry / R_mod
                            Uz_R = Rz / R_mod
                            
                            dot_prod_Reflect_Camera = Ux_R*Ux_C + Uy_R*Uy_C + Uz_R*Uz_C
                            pixel_intensity += light_intesity * math.pow(abs(dot_prod_Reflect_Camera), s)
                    
                        
                    
            elif light_matrix[n, 3] == 2: # LLum ambiental
                
                pixel_intensity += light_matrix[n, 4]

            
        if pixel_intensity > 1:
            pixel_intensity = 1

        image[Py, Px, 0] = pixel_intensity * triangle_color_matrix[nearest_triangle, 0]
        image[Py, Px, 1] = pixel_intensity * triangle_color_matrix[nearest_triangle, 1]
        image[Py, Px, 2] = pixel_intensity * triangle_color_matrix[nearest_triangle, 2]
    else:
        
        image[Py, Px, 0] = background_color[0]
        image[Py, Px, 1] = background_color[1]
        image[Py, Px, 2] = background_color[2]
                

# ////////////////////////////////////// PRINT /////////////////////////////////////////////

# Configuración del kernel
threadsperblock = (16, 16)
blockspergrid_x = math.ceil(img_x / threadsperblock[0])
blockspergrid_y = math.ceil(img_y / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)


# Copiar a memoria GPU
image_device = cuda.to_device(image)
light_matrix_device = cuda.to_device(light_matrix)
triangle_matrix_device = cuda.to_device(triangle_matrix)
changeBase_matrix_device = cuda.to_device(changeBase_matrix)
triangle_plane_matrix_device = cuda.to_device(triangle_plane_matrix)
triangle_color_matrix_device = cuda.to_device(triangle_color_matrix)
triangle_specular_device = cuda.to_device(triangle_specular)
triangle_reflectance_device = cuda.to_device(triangle_reflectance)

_input_ = (image_device,
           light_matrix_device,
           triangle_matrix_device,
           changeBase_matrix_device,
           triangle_plane_matrix_device,
           triangle_color_matrix_device,
           triangle_specular_device,
           triangle_reflectance_device)


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

# glfw.swap_interval(0)  # Deshabilita V-Sync - Si no es limiten els FPS a 60 Hz

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
    
    # Actualitzem les dades de la camera
    screen_point = cuda.to_device(Camera.punt_pla)
    base_x = cuda.to_device(Camera.coordX_pantalla)
    base_y = cuda.to_device(Camera.coordY_pantalla)
    camera_pos = cuda.to_device(Camera.camera_pos)
    
    # Generem la nova imatge
    print_image[blockspergrid, threadsperblock](*_input_, screen_point, base_x, base_y, camera_pos)
    image = image_device.copy_to_host()
    
    # Apliquem la nova imatge a la textura
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.shape[1], image.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, image)

    # Actualitzem la textura
    display_image()

    # Actualitzem la finestra - Carrega la textura que esta en el buffer de la pantalla
    glfw.swap_buffers(window)

# Terminar la aplicación
glfw.destroy_window(window)
glfw.terminate()




