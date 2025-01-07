import numpy as np
        
class camera():
    def __init__(self, camera_pos, d, pixels_x, pixels_y, len_x, len_y):
        self.camera_pos = np.array(camera_pos, dtype = float)
        self.d = d
        
        # La camara inicialment mira a la direccio [1,0,0], amb els vectors
        # que defineixen el sistema de coordenades de la pantalla a [0,0,1] i [0,1,0]
        
        self.vect_direct_cam = np.array([1,0,0]) # Recordem que aquest vector tambe correspon a l'eix Z de la camera
        self.coordY_pantalla = np.array([0,0,1]) # Recordem que aquest vector tambe correspon a l'eix Y de la camera
        self.coordX_pantalla = np.array([0,-1,0]) # Recordem que aquest vector tambe correspon a l'eix X de la camera
        
        # Definim el punt on tindrem el pla de la pantalla
        self.punt_pla = self.camera_pos + self.d * self.vect_direct_cam
        
        # Construim el pla de la pantalla
        D = -(self.vect_direct_cam[0]*self.punt_pla[0] + self.vect_direct_cam[1]*self.punt_pla[1] + self.vect_direct_cam[2]*self.punt_pla[2])
        self.pla_camera = np.array([self.vect_direct_cam[0],self.vect_direct_cam[1],self.vect_direct_cam[2], D])
        
        # Definim la pantalla
        self.pixels_x = pixels_x
        self.pixels_y = pixels_y
        self.len_x = len_x
        self.len_y = len_y
        self.mida_pixel_x = len_x/pixels_x
        self.mida_pixel_y = len_y/pixels_y
        
        self.changeBaseMatrix = np.array([self.coordX_pantalla, self.coordY_pantalla, self.vect_direct_cam], dtype = np.float64)

        
    def rotateCamara(self,theta, phi):
        # Efectuem la rotacio a la direccio que volem mirar
        self.vect_direct_cam = esfericRotate(self.vect_direct_cam, theta, phi)
        self.coordX_pantalla = esfericRotate(self.coordX_pantalla, theta, phi)
        self.coordY_pantalla = esfericRotate(self.coordY_pantalla, theta, phi)
        
        # Definim el punt on tindrem el pla de la pantalla
        self.punt_pla = self.camera_pos + self.d * self.vect_direct_cam
        
        # Construim el pla de la pantalla
        D = -(self.vect_direct_cam[0]*self.punt_pla[0] + self.vect_direct_cam[1]*self.punt_pla[1] + self.vect_direct_cam[2]*self.punt_pla[2])
        self.pla_camera = np.array([self.vect_direct_cam[0],self.vect_direct_cam[1],self.vect_direct_cam[2], D])
        
        self.changeBaseMatrix[0] = self.coordX_pantalla
        self.changeBaseMatrix[1] = self.coordY_pantalla
        self.changeBaseMatrix[2] = self.vect_direct_cam

        
    def moveCamara_rotation(self, vect, theta):
        self.camera_pos = rotateAxis(self.camera_pos, vect, theta)
        
        # Efectuem la rotacio a la direccio que volem mirar
        self.vect_direct_cam = rotateAxis(self.vect_direct_cam, vect, theta)
        self.coordX_pantalla = rotateAxis(self.coordX_pantalla, vect, theta)
        self.coordY_pantalla = rotateAxis(self.coordY_pantalla, vect, theta)
        
        # Definim el punt on tindrem el pla de la pantalla
        self.punt_pla = self.camera_pos + self.d * self.vect_direct_cam
        
        # Construim el pla de la pantalla
        D = -(self.vect_direct_cam[0]*self.punt_pla[0] + self.vect_direct_cam[1]*self.punt_pla[1] + self.vect_direct_cam[2]*self.punt_pla[2])
        self.pla_camera = np.array([self.vect_direct_cam[0],self.vect_direct_cam[1],self.vect_direct_cam[2], D])
        
        self.changeBaseMatrix[0] = self.coordX_pantalla
        self.changeBaseMatrix[1] = self.coordY_pantalla
        self.changeBaseMatrix[2] = self.vect_direct_cam
        
        
    def rotateScreenX(self, theta):
        
        # Efectuem la rotacio a l'eix X
        
        self.vect_direct_cam = rotateAxis(self.vect_direct_cam, self.coordX_pantalla, theta)
        self.coordX_pantalla = rotateAxis(self.coordX_pantalla, self.coordX_pantalla, theta)
        self.coordY_pantalla = rotateAxis(self.coordY_pantalla, self.coordX_pantalla, theta)
        
        # Definim el punt on tindrem el pla de la pantalla
        self.punt_pla = self.camera_pos + self.d * self.vect_direct_cam
        
        # Construim el pla de la pantalla
        D = -(self.vect_direct_cam[0]*self.punt_pla[0] + self.vect_direct_cam[1]*self.punt_pla[1] + self.vect_direct_cam[2]*self.punt_pla[2])
        self.pla_camera = np.array([self.vect_direct_cam[0],self.vect_direct_cam[1],self.vect_direct_cam[2], D])
        
        self.changeBaseMatrix[0] = self.coordX_pantalla
        self.changeBaseMatrix[1] = self.coordY_pantalla
        self.changeBaseMatrix[2] = self.vect_direct_cam
        
    def rotateScreenY(self, theta):
        
        # Efectuem la rotacio a l'eix Y
        
        self.vect_direct_cam = rotateAxis(self.vect_direct_cam, self.coordY_pantalla, theta)
        self.coordX_pantalla = rotateAxis(self.coordX_pantalla, self.coordY_pantalla, theta)
        self.coordY_pantalla = rotateAxis(self.coordY_pantalla, self.coordY_pantalla, theta)
        
        # Definim el punt on tindrem el pla de la pantalla
        self.punt_pla = self.camera_pos + self.d * self.vect_direct_cam
        
        # Construim el pla de la pantalla
        D = -(self.vect_direct_cam[0]*self.punt_pla[0] + self.vect_direct_cam[1]*self.punt_pla[1] + self.vect_direct_cam[2]*self.punt_pla[2])
        self.pla_camera = np.array([self.vect_direct_cam[0],self.vect_direct_cam[1],self.vect_direct_cam[2], D])
        
        self.changeBaseMatrix[0] = self.coordX_pantalla
        self.changeBaseMatrix[1] = self.coordY_pantalla
        self.changeBaseMatrix[2] = self.vect_direct_cam
        
        
    def followPoint(self, punt):
        punt = np.array(punt)
        
        # Situem el punt respecte la camera i calculem les seves coordenades esferiques
        punt = punt - self.camera_pos
        punt_esferiques = CartesianaToEsferica(punt)
        theta = punt_esferiques[1]
        phi = punt_esferiques[2]

        # Efectuem la rotacio a la direccio que volem mirar
        self.vect_direct_cam = esfericRotate(np.array([1,0,0]), theta - np.pi/2, phi)
        self.coordY_pantalla = esfericRotate(np.array([0,0,1]), theta - np.pi/2, phi)
        self.coordX_pantalla = esfericRotate(np.array([0,-1,0]), theta - np.pi/2, phi)
        
        # Definim el punt on tindrem el pla de la pantalla
        self.punt_pla = self.camera_pos + self.d * self.vect_direct_cam
        
        # Construim el pla de la pantalla
        D = -(self.vect_direct_cam[0]*self.punt_pla[0] + self.vect_direct_cam[1]*self.punt_pla[1] + self.vect_direct_cam[2]*self.punt_pla[2])
        self.pla_camera = np.array([self.vect_direct_cam[0],self.vect_direct_cam[1],self.vect_direct_cam[2], D])
        
        self.changeBaseMatrix[0] = self.coordX_pantalla
        self.changeBaseMatrix[1] = self.coordY_pantalla
        self.changeBaseMatrix[2] = self.vect_direct_cam
    
        
    def intrinsicRotation(self, theta):
        self.coordX_pantalla = rotateAxis(self.coordX_pantalla, self.vect_direct_cam, theta)
        self.coordY_pantalla = rotateAxis(self.coordY_pantalla, self.vect_direct_cam, theta)
        
        self.changeBaseMatrix[0] = self.coordX_pantalla
        self.changeBaseMatrix[1] = self.coordY_pantalla
        
        
    def moveCamera(self, vect):
        inc_y = vect[0]
        inc_x = -vect[1]
        inc_z = vect[2]
        
        # Translacionem la camara
        
        vector_translacio = inc_x*self.coordX_pantalla + inc_y*self.coordY_pantalla + inc_z*self.vect_direct_cam

        self.camera_pos += vector_translacio
            
        # Definim el punt on tindrem el pla de la pantalla
        self.punt_pla = self.camera_pos + self.d * self.vect_direct_cam
        
        # Construim el pla de la pantalla
        D = -(self.vect_direct_cam[0]*self.punt_pla[0] + self.vect_direct_cam[1]*self.punt_pla[1] + self.vect_direct_cam[2]*self.punt_pla[2])
        self.pla_camera = np.array([self.vect_direct_cam[0],self.vect_direct_cam[1],self.vect_direct_cam[2], D])
        
        

         
def calcular_interseccio_2_plans(pla1, pla2):
    # Coeficientes de los dos planos
    A1, B1, C1, D1 = pla1.A, pla1.B, pla1.C, pla1.D
    # Plano 2: A2*x + B2*y + C2*z + D2 = 0
    A2, B2, C2, D2 = pla2.A, pla2.B, pla2.C, pla2.D

    # Construcción de la matriz de coeficientes
    coeficientes = np.array([[A1, B1, C1], [A2, B2, C2]])
    
    # Construcción del vector de términos independientes
    terminos_independientes = np.array([-D1, -D2])
    
    # Encontramos un vector director de la línea de intersección usando el producto cruzado
    vector_director = np.cross([A1, B1, C1], [A2, B2, C2])
    
    if np.linalg.norm(vector_director) == 0:
        print("Plans paralels")
        return None, None
    
    else:
        
        coeficientes_XY = coeficientes[:, [0,1]]
        coeficientes_YZ = coeficientes[:, [1,2]]
        coeficientes_XZ = coeficientes[:, [0,2]]
        
        coeficient_matrix = np.empty((2,2,3))
        
        coeficient_matrix[:,:,0] = coeficientes_XY
        coeficient_matrix[:,:,1] = coeficientes_YZ
        coeficient_matrix[:,:,2] = coeficientes_XZ
        
        for n in range(3):
            try:
                # Resolver el sistema para encontrar un punto en la línea. Com no sabem quin eix talla la recta, podem tenir casos de sistemes sense solucio, provarem els tres eixos.
                # La equacio que tindrem es de la forma Ax = B
                # A es una matriu 2x2 i B es un vector de 2 dimensions
                solucion = np.linalg.solve(coeficient_matrix[:,:,n], terminos_independientes)
                existeix_solucio = True
                
            except np.linalg.LinAlgError:
                existeix_solucio = False
                
            if existeix_solucio:
                
                if n == 0: # coeficientes_XY, z = 0
                    punto = np.array([solucion[0], solucion[1], 0])
                    
                elif n == 1: # coeficientes_YZ, x = 0
                    punto = np.array([0, solucion[0], solucion[1]])
                    
                elif n == 2: # coeficientes_XZ, y = 0
                    punto = np.array([solucion[0], 0, solucion[1]])
                    
                break
            
        return punto, vector_director
                
    

def calcular_interseccion_2_rectes(recta1, recta2):
    """
    Calcula el punto de intersección entre dos rectas en 3D dadas por sus
    puntos iniciales y vectores de dirección.
    
    Parámetros:
    - p1: Punto en la recta 1 (lista o array de 3 elementos [x1, y1, z1]).
    - d1: Vector de dirección de la recta 1 (lista o array de 3 elementos [dx1, dy1, dz1]).
    - p2: Punto en la recta 2 (lista o array de 3 elementos [x2, y2, z2]).
    - d2: Vector de dirección de la recta 2 (lista o array de 3 elementos [dx2, dy2, dz2]).
    
    Retorna:
    - El punto de intersección si existe, o None si las rectas no se intersectan.
    """
    
    p1 = np.array([recta1.x0, recta1.y0, recta1.z0])
    d1 = np.array([recta1.dx, recta1.dy, recta1.dz])
    
    p2 = np.array([recta2.x0, recta2.y0, recta2.z0])
    d2 = np.array([recta2.dx, recta2.dy, recta2.dz])
    
    
    # Convertimos a matrices para resolver el sistema lineal
    A = np.array([
        [d1[0], -d2[0]],
        [d1[1], -d2[1]],
        [d1[2], -d2[2]]
    ])
    
    b = np.array([
        p2[0] - p1[0],
        p2[1] - p1[1],
        p2[2] - p1[2]
    ])
    
    # Intentar resolver el sistema A * [t, s] = b
    try:
        # Utilizamos la pseudoinversa para manejar casos no invertibles
        ts = np.linalg.lstsq(A.T @ A, A.T @ b, rcond=None)[0]
        t, s = ts
        
        # Calculamos el punto de intersección usando t
        interseccion = p1 + t * np.array(d1)
        
        # Verificamos si el punto está realmente en ambas rectas
        punto_recta_2 = p2 + s * np.array(d2)
        
        if np.allclose(interseccion, punto_recta_2, atol=1e-6):
            return interseccion
        else:
            return None
    except np.linalg.LinAlgError:
        # Si no se puede resolver el sistema, no hay intersección
        return None


def R(vect, theta):
    vect = np.array(vect)
    vect = vect / np.linalg.norm(vect) # El fem unitari
    Vx = vect[0]
    Vy = vect[1]
    Vz = vect[2]
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    fila_1 = np.array([cos + Vx*Vx*(1-cos), Vx*Vy*(1-cos) - Vz*sin, Vx*Vz*(1-cos) + Vy*sin])
    fila_2 = np.array([Vy*Vx*(1-cos) + Vz*sin, cos + Vy*Vy*(1-cos), Vy*Vz*(1-cos) - Vx*sin])
    fila_3 = np.array([Vz*Vx*(1-cos) - Vy*sin, Vz*Vy*(1-cos) + Vx*sin, cos + Vz*Vz*(1-cos)])
    
    return np.array([fila_1, fila_2, fila_3])


def esfericRotate(vect, theta, phi):
    vect = np.array(vect)
    vect_director_rotacio_eix_theta = np.array([-np.sin(phi), np.cos(phi), 0])
    R_phi = R([0,0,1], phi)
    R_theta = R(vect_director_rotacio_eix_theta, theta)
    vect = np.dot(R_phi, vect)
    vect = np.dot(R_theta, vect)
    return vect


def rotateAxis(point, vect, theta):
    vect = np.array(vect)
    R_matrix = R(vect, theta)
    return np.dot(R_matrix, point)


def CartesianaToEsferica(punt):
    
    punt = np.array(punt)
    
    x = punt[0]
    y = punt[1]
    z = punt[2]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    if z > 0:
        theta = np.arctan(np.sqrt(x**2 + y**2) / z)
    elif z == 0:
        theta = np.pi/2
    else: # z < 0 
        theta = np.pi + np.arctan(np.sqrt(x**2 + y**2) / z)
        
    if x > 0 and y > 0:
        phi = np.arctan(y/x)
    elif x > 0 and y < 0:
        phi = 2*np.pi + np.arctan(y/x)
    elif x == 0:
        phi = np.pi/2 * np.sign(y)
    else: # x < 0:
        phi = np.pi + np.arctan(y/x)
        
    return np.array([r, theta, phi])


def EsfericaToCartesiana(punt):
    r = punt[0]
    theta = punt[1]
    phi = punt[2]
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.array([x,y,z])
