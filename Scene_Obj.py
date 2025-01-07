import numpy as np

class Triangle:
    
    def __init__(self, vertex0, vertex1, vertex2, color, specular, reflectance):
        
        self.vertexs = np.array([vertex0, vertex1, vertex2], dtype = np.float64)
        self.triangles = np.array([[0,1,2]], dtype = np.uint16)
        self.color = np.array(color, dtype = np.uint8)
        self.specular = specular
        self.reflectance = reflectance
        
        self.number_vertexs = 3
        self.number_triangles = 1
        
class Rectangle:
    
    def __init__(self, vertex0, vertex1, vertex2, color, specular, reflectance):
        
        vertex0 = np.array(vertex0)
        vertex1 = np.array(vertex1)
        vertex2 = np.array(vertex2)
        vertex3 = vertex1 + vertex2 - vertex0
        
        self.vertexs = np.array([vertex0, vertex1, vertex2, vertex3], dtype = np.float64)
        self.triangles = np.array([[0, 1, 2], [3, 2, 1]], dtype = np.uint16)
        self.color = np.array(color, dtype = np.uint8)
        self.specular = specular
        self.reflectance = reflectance
        
        self.number_vertexs = 4
        self.number_triangles = 2
        
class RectanglePrisma():
    
    def __init__(self, vertex0, vertex1, vertex2, vertexH, color, specular, reflectance):
    
        vertex0 = np.array(vertex0)
        vertex1 = np.array(vertex1)
        vertex2 = np.array(vertex2)
        vertex3 = vertex1 + vertex2 - vertex0
        
        vertexH = np.array(vertexH) - vertex0
        
        vertex4 = vertexH + vertex0 #0H
        vertex5 = vertexH + vertex1 #1H
        vertex6 = vertexH + vertex2 #2H
        vertex7 = vertexH + vertex3 #3H

        self.vertexs = np.array([vertex0, vertex1, vertex2, vertex3, vertex4, vertex5, vertex6, vertex7], dtype = np.float64)
        self.triangles = np.array([[0, 1, 2], [4, 6, 5],[0, 4, 1], [0, 2, 4],[1, 5, 3], [2, 3, 6],[3, 2, 1], [7, 5, 6],[1, 4, 5], [4, 2, 6],[3, 5, 7], [3, 7, 6]], dtype = np.uint16)
        self.color = np.array(color, dtype = np.uint8)
        self.specular = specular
        self.reflectance = reflectance
        
        self.number_vertexs = 8
        self.number_triangles = 12

    
def normalizar_vector(v):
    """Normaliza un vector para que tenga longitud 1."""
    return v / np.linalg.norm(v)

def crear_icosaedro(radio):
    """Crea un icosaedro inicial."""
    # Ángulo dorado para distribuir vértices
    phi = (1 + np.sqrt(5)) / 2

    # Coordenadas iniciales del icosaedro (sin escalado)
    vertices = [
        (-1,  phi,  0), ( 1,  phi,  0), (-1, -phi,  0), ( 1, -phi,  0),
        ( 0, -1,  phi), ( 0,  1,  phi), ( 0, -1, -phi), ( 0,  1, -phi),
        ( phi,  0, -1), ( phi,  0,  1), (-phi,  0, -1), (-phi,  0,  1),
    ]

    # Normalizar los vértices y escalarlos al radio de la esfera
    vertices = [normalizar_vector(np.array(v)) * radio for v in vertices]

    # Caras del icosaedro (triángulos)
    indices = [
        (0, 11,  5), (0,  5,  1), (0,  1,  7), (0,  7, 10), (0, 10, 11),
        (1,  5,  9), (5, 11,  4), (11, 10,  2), (10,  7,  6), (7,  1,  8),
        (3,  9,  4), (3,  4,  2), (3,  2,  6), (3,  6,  8), (3,  8,  9),
        (4,  9,  5), (2,  4, 11), (6,  2, 10), (8,  6,  7), (9,  8,  1),
    ]

    return vertices, indices

def subdividir(vertices, indices, radio):
    """Subdivide cada triángulo en 4 triángulos más pequeños."""
    nuevo_vertices = vertices[:]
    nuevo_indices = []
    midpoint_cache = {}

    def obtener_punto_medio(v1, v2):
        """Encuentra el punto medio entre dos vértices y lo normaliza."""
        menor, mayor = sorted((v1, v2))  # Ordenar para evitar duplicados
        if (menor, mayor) in midpoint_cache:
            return midpoint_cache[(menor, mayor)]
        
        punto_medio = normalizar_vector((np.array(nuevo_vertices[menor]) + np.array(nuevo_vertices[mayor])) / 2) * radio
        nuevo_vertices.append(punto_medio)
        midpoint_cache[(menor, mayor)] = len(nuevo_vertices) - 1
        return midpoint_cache[(menor, mayor)]

    for tri in indices:
        v1, v2, v3 = tri
        a = obtener_punto_medio(v1, v2)
        b = obtener_punto_medio(v2, v3)
        c = obtener_punto_medio(v3, v1)

        nuevo_indices.extend([
            (v1, a, c),
            (v2, b, a),
            (v3, c, b),
            (a, b, c),
        ])

    return nuevo_vertices, nuevo_indices

def generar_esfera_icosaedro(radio, subdivisiones):
    """Genera una esfera basada en un icosaedro y subdivisiones."""
    vertices, indices = crear_icosaedro(radio)
    for _ in range(subdivisiones):
        vertices, indices = subdividir(vertices, indices, radio)
    return vertices, indices


class Scene:

    def __init__(self):

        self.instances = list()
        self.lights = list()
        self.total_vertexs = 0  
        self.total_triangles = 0

    def addInstance(self, object_instance):

        self.instances.append(object_instance)
        self.total_vertexs += object_instance.number_vertexs
        self.total_triangles += object_instance.number_triangles


        
    # def addSphere(self, position, radio, subdivisiones, color, specular, reflectance):
        
    #     position = np.array(position)
    #     color = np.array(color) 

    #     # Generar la esfera
    #     vertices, indices = generar_esfera_icosaedro(radio, subdivisiones)
        
    #     for _triangle_ in indices:

    #         # Fem la translacio dels vertexs, ja que la esfera es genera en [0, 0, 0]
    #         vertex_x = vertices[_triangle_[0]] + position
    #         vertex_y = vertices[_triangle_[1]] + position
    #         vertex_z = vertices[_triangle_[2]] + position

    #         self.addTriangle(vertex_x, vertex_y, vertex_z, color, specular, reflectance)

              
    def addLight(self, parameter, type_light, intensity):

        if type_light != "puntual" and type_light != "directional" and type_light != "ambiental":
            raise Exception("Invalid type of light")

        light_dict = dict()
        light_dict["parameter"] = parameter
        light_dict["type"] = type_light
        light_dict["intensity"] = intensity
        
        
        self.lights.append(light_dict)
    
   

